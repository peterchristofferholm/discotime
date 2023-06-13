import logging

from pytest import fixture
from pytest import mark
from lightning import Trainer

from discotime.datasets import Mgus2, DataConfig
from discotime.models import LitSurvModule, ModelConfig

# configure logging at the root level of Lightning
logging.getLogger("lightning").setLevel(logging.ERROR)


@fixture
def mgus2_dm():
    dm = Mgus2(
        DataConfig(
            batch_size=60, n_time_bins=20, discretization_scheme="number"
        )
    )
    dm.prepare_data()
    return dm


def test_data_setup(mgus2_dm: Mgus2):
    mgus2_dm.setup(stage="fit")
    assert mgus2_dm.time_range == (2.0, 373.0)

    # hparams should be accessible
    assert set(mgus2_dm.hparams.keys()) == {"data_config", "seed"}

    assert len(mgus2_dm.dset_train[0]) == 5
    assert len(mgus2_dm.dset_test[0]) == 5

    assert len(next(iter(mgus2_dm.train_dataloader()))) == 5
    assert len(next(iter(mgus2_dm.test_dataloader()))) == 5

    # number of cuts is n_time_bins + 1
    assert len(mgus2_dm.cuts) == (mgus2_dm.hparams.data_config.n_time_bins + 1)  # type: ignore


@mark.filterwarnings("ignore")
def test_model_integration(mgus2_dm: Mgus2):
    model = LitSurvModule(ModelConfig())
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=20,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, mgus2_dm)
    trainer.test(model, mgus2_dm)

    assert True
