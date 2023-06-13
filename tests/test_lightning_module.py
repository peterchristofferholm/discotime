import pytest
from lightning.pytorch import Trainer
from warnings import filterwarnings
from pathlib import Path

from discotime.models import ModelConfig, LitSurvModule
from discotime.datasets import DataConfig, Mgus2


@pytest.fixture
def lit_data_module():
    return Mgus2(DataConfig())


@pytest.fixture
def lit_module():
    return LitSurvModule(ModelConfig(n_sequential_blocks=10))


def test_load_checkpoint_works(
    lit_module: LitSurvModule, lit_data_module: Mgus2, tmp_path: Path
):
    filterwarnings("ignore", message=".*num_workers")
    filterwarnings("ignore", message=".*self.log")

    trainer = Trainer(barebones=True, max_epochs=1)
    trainer.fit(lit_module, lit_data_module)
    trainer.save_checkpoint(tmp_path / "checkpoint.ckpt")

    # this needs to work without any errors
    lit_module = lit_module.__class__.load_from_checkpoint(
        checkpoint_path=tmp_path / "checkpoint.ckpt", strict=True
    )
