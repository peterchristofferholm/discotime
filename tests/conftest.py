import pytest

from discotime.models import LitSurvModule, ModelConfig
from discotime.datasets import Mgus2, DataConfig


@pytest.fixture(scope="session")
def survival_data_1():
    """Example data from p. 18 in Kleinbaum & Klein (2005)"""
    data = {
        1: [6, 6, 6, 7, 10, 13, 16, 22, 23],
        0: [6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35],
    }
    time, event = zip(*[(t, e) for (e, ts) in data.items() for t in ts])
    return (time, event)


@pytest.fixture(scope="session")
def survival_data_2():
    """Example data from Chap. 9 in Kleinbaum & Klein (2005)."""

    # [[censored], [cause_1], [cause_2]]
    data = [
        [3.2, 7.6, 10, 11, 15, 24.4],
        [0.7, 3, 4.9, 6, 6, 6.9, 10, 10.8, 17.1, 20.3],
        [1.5, 2.8, 3.8, 4.7, 7, 10, 10, 11.2],
    ]
    time, event = zip(
        *[(v, i) for (i, times) in enumerate(data) for v in times]
    )
    return (time, event)


@pytest.fixture(scope="session")
def default_mgus2_model():
    model = LitSurvModule(ModelConfig())
    model.datamodule = Mgus2(DataConfig(batch_size=64))
    model.datamodule.setup(stage="fit")
    model.setup(stage="fit")
    return model
