import warnings
import numpy as np
import pytest

from discotime.datasets.utils import LabelDiscretizer
from discotime.datasets.utils import (
    _discretize_labels,
    _cuts_number,
)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ((0.1, 0), (0, 0)),
        ((0.6, 0), (1, 0)),
        ((0.6, 1), (0, 1)),
        ((1.6, 2), (1, 2)),
        ((2.1, 0), (2, 0)),
        ((3.1, 1), (2, 0)),
        ((3.8, 2), (2, 0)),
    ],
)
def test_label_discretization(input, expected):
    cuts = np.array([0, 1, 2, 3])
    assert _discretize_labels(input[0], input[1], cuts=cuts) == expected


def test_discretization_scheme_number():
    time = [1, 2, 3, 4, 5]
    event = [1, 1, 1, 1, 1]
    expected = [0, 1, 2, 3, 4, 5]
    cuts = _cuts_number(n_bins=5, max_time=5, time=time, event=event)
    assert cuts == pytest.approx(expected)


class TestLabTransDiscrete:
    def test_interval_discretization(self, comprisk_testdata):
        time, event = comprisk_testdata
        ltd = LabelDiscretizer("interval", 10)
        dt, de = ltd.fit_transform(time, event)
        assert len(set(dt)) == 10
        assert len(set(np.round(np.diff(ltd.cuts), 4))) == 1
        assert len(set(map(lambda x: x.size, (time, event, dt, de)))) == 1

    def test_quantile_discretization(self, comprisk_testdata):
        time, event = comprisk_testdata
        ltd = LabelDiscretizer("number", 10)
        dt, de = ltd.fit_transform(time, event)
        assert len(set(dt)) == 10
        assert len(set(np.diff(ltd.cuts))) != 1
        assert len(set(map(lambda x: x.size, (time, event, dt, de)))) == 1

    def test_manual_discretization(self, comprisk_testdata):
        time, event = comprisk_testdata
        ltd = LabelDiscretizer(cut_points=[0, 5, 10, 15])
        with pytest.warns(UserWarning, match="ignoring fit()"):
            dt, de = ltd.fit_transform(time, event)
        assert len(set(dt)) == 3
        assert len(set(map(lambda x: x.size, (time, event, dt, de)))) == 1
