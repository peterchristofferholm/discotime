from math import inf
from collections import deque, defaultdict
from collections.abc import Mapping
from typing import Generator, TypeVar, Optional
from numbers import Number

import lightning.pytorch as pl
from lightning.pytorch import cli
from optuna.trial import Trial
from optuna.exceptions import TrialPruned

from lightning.pytorch.callbacks import Callback
import lightning as L
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
    ConcatDataset,
    Subset,
)

from discotime.models import LitSurvModule
from discotime.datasets.utils import SurvDataset

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def update_mapping(d: Mapping, u: Mapping) -> Mapping:
    """Update nested mappings recursively.

    Args:
        d (Mapping): dictionary/mapping of things.
        u (Mapping): other dictionary/mapping of things to update `d` with.

    Example:
        >>> foo = {"dog" : {"color" : "black", "age" : 10}}
        >>> bar = {"dog" : {"age" : 11}, "cat" : None}
        >>> update_mapping(foo, bar)
        {'dog': {'color': 'black', 'age': 11}, 'cat': None}
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_mapping(d.get(k, {}), v)  # type: ignore
        else:
            d[k] = v  # type: ignore
    return d


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


class BaseFabricTrainer:
    """Bare-bones trainer class using Fabric.

    This trainer class supports only a very limited set of the full lightning
    functionality. It can be useful for quick prototyping or for k-fold
    cross-validation as exemplified in the scripts that can be found in the
    ``/experiments`` folder on github.
    """

    def __init__(
        self,
        module: LitSurvModule,
        dset_train: SurvDataset,
        dset_valid: SurvDataset,
        batch_size: int,
    ) -> None:
        self.fabric = L.Fabric(accelerator="cpu", devices=1)
        self.train_dl = DataLoader(
            dset_train, shuffle=True, batch_size=batch_size, drop_last=True
        )
        self.valid_dl = DataLoader(
            dset_valid, shuffle=True, batch_size=batch_size
        )
        self.module = module

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        self.fabric.launch()
        train_dl = self.fabric.setup_dataloaders(self.train_dl)
        valid_dl = self.fabric.setup_dataloaders(self.valid_dl)
        module, optimizer = self.fabric.setup(
            self.module,
            self.module.configure_optimizers(),
        )

        while True:
            self.module.train()
            with torch.set_grad_enabled(True):
                for batch_idx, batch in enumerate(train_dl):
                    loss = module.training_step(batch, batch_idx)
                    optimizer.zero_grad()
                    self.fabric.backward(loss)
                    optimizer.step()

            with torch.set_grad_enabled(False):
                module.eval()
                losses = (
                    module.validation_step(batch, idx) * len(batch[1])
                    for idx, batch in enumerate(valid_dl)
                )
                yield sum(losses) / len(valid_dl.sampler)


def split_dataset(
    dataset: Dataset[T], k: int = 5
) -> Generator[tuple[ConcatDataset[T], Subset[T]], None, None]:
    """Split dataset into k train/validation splits"""
    splits = deque(random_split(dataset, lengths=[1 / k] * k))
    for _ in range(k):
        dset_valid: Subset[T] = splits.popleft()
        dset_train: ConcatDataset[T] = ConcatDataset(splits)
        yield (dset_train, dset_valid)
        splits.append(dset_valid)


class EarlyStopping:
    """Lightning-free implementation of early stopping.

    To be used with the bare-bones simple trainer that have no support for
    lightning callbacks.
    """

    def __init__(
        self, patience: int, min_delta: float, direction: str = "minimize"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        if direction not in (allowed := {"minimize", "maximize"}):
            raise ValueError(f"direction can be {allowed}, got {direction}")

        self.direction = direction
        self.best = inf if direction == "minimize" else -inf

    def should_stop(self, metric) -> bool:
        match self.direction:
            case "minimize":
                improvement = metric < (self.best - self.min_delta)
            case "maximize":
                improvement = metric > (self.best + self.min_delta)

        if improvement:
            self.wait = 0
            self.best = metric
        else:
            self.wait += 1

        return self.wait >= self.patience


def get_last_intermediate_value(trial: Trial):
    values = trial.storage.get_trial(trial._trial_id).intermediate_values
    return max(values.items(), key=lambda x: x[0])[1]


def clamp(value: float, minvalue: float, maxvalue: float):
    return max(minvalue, min(value, maxvalue))


class OptunaCallback(Callback):
    """Lightning callback for hyperparameter optimization with Optuna.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current
            evaluation of the objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``.
    """

    def __init__(
        self,
        trial: Trial,
        monitor: str,
        minvalue: float = -inf,
        maxvalue: float = inf,
    ) -> None:
        super().__init__()

        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(
        self, trainer: pl.Trainer, module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            raise ValueError(
                f"The metric '{self.monitor}' is not in the evaluation logs."
                f"Available metrics are: {trainer.callback_metrics}"
            )

        epoch = module.current_epoch
        self.trial.report(
            clamp(metric.item(), self.minvalue, self.maxvalue),
            step=epoch,
        )

        if not self.trial.should_prune():
            return

        raise TrialPruned(f"Trial was pruned at epoch {epoch}.")
