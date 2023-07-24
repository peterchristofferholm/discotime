from typing import Optional, Any, Callable, Dict, Sequence
from dataclasses import dataclass
import warnings

import torch
from torch.nn import functional as F
from torch import nn
from lightning import pytorch as pl

from discotime.models.components import Net, negative_log_likelihood
from discotime.metrics import BrierScoreScaled
from discotime.datasets import LitSurvDataModule
from discotime.utils import Interpolate2D


@dataclass(kw_only=True)
class ModelConfig:
    learning_rate: float = 1e-3
    activation_function: str = "SiLU"
    """Name of activation function (str). The name needs to match one of the
    activation functions in the ``torch.nn`` module.
    """

    n_sequential_blocks: int = 2
    """Number of neural-network layer blocks.

    Each block is modelled after a ResNet skip-block and contains the following
    key elements:

    .. code-block ::

        Sequential(
            (0): LazyBatchNorm1d()
            (1): LazyLinear()
            (2): DropOut()
            (3): SiLU()
            (4): LazyLinear()
            (5): DropOut()
        )

    Per default the output of the model is ``self.act_fn(x + self.net(x))``
    but the skip part can be removed by setting :attr:`use_skip_connections` to
    ``False``. :attr:`n_hidden_units` controls the size of the linear layers in
    each block. See :class:`.components.Block` for more details
    on the actual implementation.
    """
    n_hidden_units: int = 20
    """Number of neurons in each of the hidden layers.

    For ease of of use, the size of each hidden layer have been constrained to
    to be the exact same size as all the others. Default is 20.
    """
    dropout_rate: Optional[float] = None
    """Should ``nn.Dropout()`` be used in each block, and if so what is the
    rate of dropout? If ``None`` then dropout is not used. Default is None.
    """
    batch_normalization: bool = True
    """Use batch normalization in each block? Default is True."""
    use_skip_connections: bool = False
    """Toggle the use of skip-connections in the model blocks."""

    evaluation_grid_size: Optional[int] = 50
    """An integer `n`, then `n` evenly
    distributed timepoint are chosen from the range of the data.
    """
    evaluation_grid_cuts: Optional[Sequence[float]] = None
    """Specify the grid (seq of floats) at which model metrics are calculated.
    """

    n_time_bins: Optional[int] = None
    """Number of time bins in discretization grid.

    If ``None``, which is the default, then we try to get the information from
    the attached datamodule during setup.
    """
    n_risks: Optional[int] = None
    """How many competing risks are we dealing with?

    If ``None``, which is the default, then we try to get the information from
    the attached datamodule during setup.
    """


class LitSurvModule(pl.LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.learning_rate = config.learning_rate

        self._loss = negative_log_likelihood
        self._metrics: dict[str, Callable] = {}
        self._eval_grid: Optional[torch.Tensor] = None
        self._data_cuts: Optional[torch.Tensor] = None
        self._model: Optional[nn.Module] = None
        self._datamodule: Optional[LitSurvDataModule] = None

        self._test_step_outputs: list[tuple[torch.Tensor, ...]] = []
        self._validation_step_outputs: list[tuple[torch.Tensor, ...]] = []

        self.save_hyperparameters()

    @property
    def datamodule(self) -> LitSurvDataModule:
        if self._datamodule is not None:
            return self._datamodule
        return self.trainer.datamodule  # type: ignore

    @datamodule.setter
    def datamodule(self, value: LitSurvDataModule) -> None:
        self._datamodule = value

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise AssertionError(
                "`self.model` have not been instantiated."
                "Try calling `self.setup()` first."
            )
        return self._model

    @property
    def eval_grid(self) -> torch.Tensor:
        if self._eval_grid is None:
            raise AssertionError(
                (
                    "`self.eval_grid` has not been instantiated."
                    "Try calling `self.setup()` first."
                )
            )
        return self._eval_grid.to(self.device)

    @property
    def data_cuts(self) -> torch.Tensor:
        if self._data_cuts is None:
            raise AssertionError(
                (
                    "`self.data_cuts` has not been instantiated."
                    "Try calling `self.setup()` first."
                )
            )
        return self._data_cuts.to(self.device)

    def setup(self, stage):
        """Called at the beginning of fit (train + validate), validate, test,
        or predict. This is where the PyTorch model gets instantiated.

        Args:
            stage (`str`): either ``"fit"``, ``"validate"`` ``"test"`` or
                ``"predict"``
        """
        if not isinstance((conf := self.config), ModelConfig):
            raise TypeError("`config` has wrong type!")

        def _get_data_attribute(attr: str) -> int:
            """Get attribute from config or attached datamodule."""
            if (value := getattr(conf, attr)) is None:
                try:
                    self.datamodule.setup()  # fit transformers
                    return getattr(self.datamodule, attr)
                except:
                    raise AssertionError(f"`{attr}` could not be found.")
            return value

        self.n_time_bins = _get_data_attribute("n_time_bins")
        self.n_risks = _get_data_attribute("n_risks")

        if not self._model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = Net(
                    n_out_features=self.n_time_bins * (self.n_risks + 1),
                    n_hidden_units=conf.n_hidden_units,
                    add_skip_connection=conf.use_skip_connections,
                    activation_function=getattr(nn, conf.activation_function),
                    batch_normalization=conf.batch_normalization,
                    dropout_rate=conf.dropout_rate,
                    n_blocks=conf.n_sequential_blocks,
                )

        if stage in {"fit", "test", "validate"}:
            self.datamodule.setup()

            # get attributes from the datamodule
            self._data_cuts = torch.as_tensor(self.datamodule.cuts)
            if conf.evaluation_grid_cuts is not None:
                self._eval_grid = torch.as_tensor(
                    conf.evaluation_grid_cuts, device=self.device
                )
            else:
                start, end = self.datamodule.time_range
                self._eval_grid = torch.linspace(
                    start=start,
                    end=end,
                    steps=conf.evaluation_grid_size,
                    device=self.device,
                )

            # instantiate metrics
            self._metrics["IPA"] = BrierScoreScaled(
                survival_train=(
                    self.datamodule.dset_fit.event_time_cont,
                    self.datamodule.dset_fit.event_status_cont,
                ),
                eval_grid=self.eval_grid,
                integrate=True,
            )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["model"] = self._model

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._model = checkpoint["model"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1, self.n_time_bins, self.n_risks + 1)

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        x, dt, de, _, _ = batch
        train_loss = self._loss(self(x), dt, de)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        x, dt, de, ct, ce = batch
        val_loss = self._loss(self(x), dt, de)
        self.log("val_loss", val_loss)
        # aggregate estimates + labels for batch metrics
        self._validation_step_outputs.append(
            (self._predict_estimates(x, self.eval_grid), ct, ce)
        )
        return val_loss

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        x, dt, de, ct, ce = batch
        test_loss = self._loss(self(x), dt, de)
        self.log("test_loss", test_loss)
        # aggregate estimates + labels for batch metrics
        self._test_step_outputs.append(
            (self._predict_estimates(x, self.eval_grid), ct, ce)
        )
        return test_loss

    def on_test_epoch_end(self) -> None:
        estimates, ct, ce = map(
            lambda x: torch.cat(x), zip(*self._test_step_outputs)
        )
        self._test_step_outputs.clear()  # empty list
        for metric, metric_fn in self._metrics.items():
            values = metric_fn(estimates, (ct, ce))
            for cause, value in enumerate(values, start=1):
                self.log(f"test_{metric}_cause{cause}", value)

    def on_validation_epoch_end(self) -> None:
        estimates, ct, ce = map(
            lambda x: torch.cat(x).cpu(), zip(*self._validation_step_outputs)
        )
        self._validation_step_outputs.clear()  # empty list
        for metric, metric_fn in self._metrics.items():
            values = metric_fn(estimates, (ct, ce))
            for cause, value in enumerate(values, start=1):
                self.log(f"val_{metric}_cause{cause}", value)

    def _predict_estimates(
        self, x: torch.Tensor, timepoints: Any
    ) -> torch.Tensor:
        est = F.softmax(self(x), dim=-1)
        surv = torch.cumprod(est[..., [0]], dim=1)

        surv_lag = torch.roll(surv, shifts=1, dims=1)
        surv_lag[:, 0, :] = 1  # starts with 100%

        # convert conditional hazards to cause-specific cumulative incidence
        proba = torch.cumsum(surv_lag * est[..., 1:], dim=1)
        proba = F.pad(proba, pad=(0, 0, 1, 0))

        return Interpolate2D(self.data_cuts, proba, dim=1)(timepoints)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), self.learning_rate)
