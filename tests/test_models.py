import pytest
from pytest import approx
from warnings import filterwarnings

from lightning import Trainer
import torch
import torch.nn.functional as F

from discotime.models import ModelConfig, LitSurvModule
from discotime.datasets import Mgus2, DataConfig
from discotime.models.components import negative_log_likelihood


def test_nll_loss():
    time = torch.tensor([0, 0, 1, 2, 3])
    event = torch.tensor([1, 2, 0, 0, 1])
    logits = torch.rand((5, 4, 3))
    phi = F.log_softmax(logits, dim=2)

    # calculate loss "manually" from the tutz formula
    loss0 = 0
    for i, (_time, _event) in enumerate(zip(time, event)):
        for t in range(_time + 1):
            if t == _time:
                loss0 -= phi[i][t][_event]
            else:
                loss0 -= phi[i][t][0]  # no event
    loss0 = loss0 / len(time)
    loss1 = negative_log_likelihood(logits, time, event)

    assert loss1 == pytest.approx(loss0, abs=1e-6)


def test_epoch_loss_batch_averaging():
    """Total epoch loss should be independent of batch size."""

    filterwarnings("ignore", message=".*`self.log")

    mgus_bs60 = Mgus2(DataConfig(batch_size=60))
    mgus_bs30 = Mgus2(DataConfig(batch_size=30))
    mgus_bs30.prepare_data()

    model = LitSurvModule(
        ModelConfig(
            n_risks=mgus_bs30.n_risks,
            n_time_bins=mgus_bs30.n_time_bins,
        )
    )
    model.datamodule = mgus_bs30
    model.setup(stage="fit")

    def get_epoch_loss(dataset):
        dataset.setup(stage="fit")
        with torch.set_grad_enabled(False):
            model.eval()
            losses = (
                model.validation_step(batch, batch_idx) * len(batch[1])
                for batch_idx, batch in enumerate(dataset.train_dataloader())
            )
            return sum(losses) / len(dataset.dset_train)

    assert get_epoch_loss(mgus_bs30) == approx(get_epoch_loss(mgus_bs60))


def test_model_output_shape():
    filterwarnings("ignore", message=".*`self.log")
    model = LitSurvModule(ModelConfig())
    model.datamodule = Mgus2(
        DataConfig(
            batch_size=60,
            n_time_bins=20,
            discretization_scheme="number",
        )
    )
    model.datamodule.setup(stage="fit")
    model.setup(stage="fit")

    X = torch.vstack(
        [torch.tensor(d.features) for d in model.datamodule.dset_fit]
    )

    n_obs, n_time_bins, n_risks = X.shape[0], model.n_time_bins, model.n_risks

    assert model(X).shape == (n_obs, n_time_bins, n_risks + 1)
    assert model.predict_step(X, 0).shape == (n_obs, n_time_bins, n_risks + 1)
    assert model._predict_estimates(X, [2, 3]).shape == (n_obs, 2, n_risks)
