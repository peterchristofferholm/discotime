import torch
import pytest
import hypothesis
import hypothesis.strategies as st
from einops import repeat

from discotime.utils import KaplanMeier, AalenJohansen, Interpolate2D

### KaplanMeier ###############################################################


@pytest.fixture
def km_estimator():
    """Example data from p. 18 in Kleinbaum & Klein (2005)"""
    data = {
        1: [6, 6, 6, 7, 10, 13, 16, 22, 23],
        0: [6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35],
    }
    time, event = zip(*[(t, e) for (e, ts) in data.items() for t in ts])
    return KaplanMeier(time, event)


@pytest.mark.parametrize(
    ("time", "expected"),
    [
        ([0], [1]),
        (0, [1]),
        ([0, 0], [1, 1]),
        (6, [0.8571]),
        (22, [0.5378]),
        ([6, 7], [0.8571, 0.8067]),
        ([7, 6], [0.8067, 0.8571]),
    ],
)
def test_kaplan_meier_estimates(time, expected, km_estimator):
    assert km_estimator(time) == pytest.approx(expected, abs=1e-4)


def test_kaplan_meier_percentiles(km_estimator):
    t = km_estimator.percentile(km_estimator._sj)
    assert all(km_estimator(t) == km_estimator._sj)


@pytest.mark.parametrize("percentile", [-1.1, 1.1])
def test_kaplan_meier_percentiles_errors(percentile, km_estimator):
    with pytest.raises(ValueError):
        km_estimator.percentile(percentile)


# km.percentile should stop giving unique timepoints if p < P(t_max)
def test_kaplan_meier_percentiles_unobserved(km_estimator):
    t_max, p_min = km_estimator._tj[-1], km_estimator._sj[-1]
    assert km_estimator.percentile(p_min - 0.05) == t_max


###############################################################################
# Aalen-Johansen estimator


@pytest.fixture
def cic(survival_data_2):
    time, event = survival_data_2
    return AalenJohansen(time, event)


@pytest.mark.parametrize(
    ("timepoint", "cause", "expected"),
    [
        # fmt: off
       # expected obtained with `prodlim` package in R
       ([10.7, 10.8, 10.9], 1, [0.3066, 0.3613, 0.3613]),
       ([ 0.0,  0.6,  0.7], 1, [0.0000, 0.0000, 0.0416]),
       ([ 0.0,  0.6,  0.7], 2, [0.0000, 0.0000, 0.0000]),
       ([17.1, 20.3, 24.4], 1, [0.4488, 0.5362, 0.5362]),
        # fmt: on
    ],
)
def test_cumulative_incidence_comprisk(timepoint, cause, expected, cic):
    estimates = cic(timepoint)[:, cause - 1].squeeze()
    assert estimates == pytest.approx(expected, abs=1e-4)


###############################################################################


@st.composite
def interpolate2d_strategy(draw):
    xp = draw(st.lists(st.integers(0, 20), min_size=2, unique=True))
    x = draw(st.lists(st.integers(min(xp), max(xp)), min_size=1, unique=True))
    coef = draw(st.tuples(st.floats(-10, 10), st.floats(0, 10)))
    return (torch.tensor(sorted(x)), torch.tensor(sorted(xp)), coef)


@hypothesis.given(interpolate2d_strategy())
@hypothesis.example(
    args=(
        torch.tensor([1.0]),
        torch.tensor([0.0000, 0.7500, 1.0000]),
        torch.tensor([1.0, 0.0]),
    )
)
def test_interpolate2d_linear(args):
    x, xp, coef = args

    hypothesis.assume(torch.unique(xp).size() == xp.size())
    hypothesis.assume(torch.unique(x).size() == x.size())

    yp = coef[0] * xp + coef[1]
    yp = repeat(yp, "t -> b r t", b=10, r=2)

    y = coef[0] * x + coef[1]
    y = repeat(y, "t -> b r t", b=10, r=2)

    assert Interpolate2D(xp, yp)(x) == pytest.approx(y, abs=1e-3)
