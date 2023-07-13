import numpy as np
import torch
import pytest

from discotime.metrics.brier_score import _cic_pv
from discotime.metrics import BrierScore, IPA


def test_cic_pseudovalues(comprisk_testdata):
    """Test _cic_pseudovalues.

    See if the implementation gives similar results to similar to jacknife
    implementation provided by the `prodlim` package in R.

    The expected output is obtained like this:

    ```{r}
    pfit <- prodlim(Hist(time, status) ~ 1, data = data)
    jackknife(pfit, times=c(0, 3.2, 10, 10.1, 20.1), cause=1)
    ```
    """
    time, event = comprisk_testdata
    tau, cause = [0, 3.2, 10, 10.1, 20.1], 1
    expected = [
        [0.0000, 0.0000, 0.2679, 0.2679, 0.4386],
        [0.0000, 0.0000, 0.0811, 0.0811, 0.3567],
        [0.0000, 0.0000, -0.0245, -0.0245, 0.3688],
        [0.0000, 0.0000, -0.0245, -0.0245, 0.2269],
        [0.0000, 0.0000, -0.0245, -0.0245, 0.3398],
        [0.0000, 0.0000, -0.0245, -0.0245, -0.5069],
        [0.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [0.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [0.0000, 0.0000, 1.0407, 1.0407, 1.0312],
        [0.0000, 0.0000, 1.0407, 1.0407, 1.0312],
        [0.0000, 0.0000, 1.0407, 1.0407, 1.0312],
        [0.0000, 0.0000, 1.0407, 1.0407, 1.0312],
        [0.0000, 0.0000, 1.1366, 1.1366, 1.0986],
        [0.0000, 0.0000, -0.0245, -0.0245, 1.2203],
        [0.0000, 0.0000, -0.0245, -0.0245, 2.0331],
        [0.0000, 0.0000, -0.0245, -0.0245, -0.5069],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, -0.0149, -0.0149, -0.0244],
        [0.0000, 0.0000, -0.0149, -0.0149, -0.0244],
        [0.0000, 0.0000, -0.0149, -0.0149, -0.0244],
        [0.0000, 0.0000, -0.0245, -0.0245, -0.0625],
        [0.0000, 0.0000, -0.0245, -0.0245, -0.0625],
        [0.0000, 0.0000, -0.0245, -0.0245, -0.2247],
    ]
    cic_pseudo = _cic_pv(time, event, np.asarray(tau))[..., cause - 1]
    assert cic_pseudo == pytest.approx(np.asarray(expected), abs=1e-4)


@pytest.mark.parametrize(
    ("cause", "timepoint", "expected"),
    [
        (1, [8.67], [0.192]),
        (1, [15.77, 1.77], [0.04, 0.231]),
        (1, [20.4, 2.57, 11.95, 12.85], [0.04, 0.231, 0.231, 0.249]),
        (1, [8.71], [0.192]),
        (1, [16.41, 20.18], [0.231, 0.247]),
        (2, [21.84, 20.91, 18.83], [0.235, 0.235, 0.235]),
        (2, [20.31, 12.49, 2.92, 0.56], [0, 0.076, 0.235, 0.235]),
        (2, [23.38, 18.46, 15.24], [0.235, 0.235, 0.235]),
    ],
)
def test_brier_score_null(timepoint, cause, expected, comprisk_testdata):
    """Expected values obtained using the `riskRegression` R package.

    ```{r}
    cfit <- CSC(Hist(t, e) ~ 1, cause = _, data = data)
    Score(list(cfit), formula=Hist(t, e) ~ 1, data=data, times=c(...), cause=_)
    ```
    """
    t, e = comprisk_testdata
    timepoint = np.sort(timepoint)

    brier_score_null, _ = BrierScore(timepoint, [cause])(None, t, e)
    assert brier_score_null == pytest.approx(expected, abs=1e-3)


def test_index_of_prediction_accuracy():
    timepoints = np.arange(5, dtype=np.float64)

    time = np.repeat(timepoints, repeats=3)
    event = np.tile(np.arange(3, dtype=np.int64), reps=5)
    phi = np.zeros((time.size, timepoints.size, 2))

    bs = BrierScore(timepoints)
    bs_n, bs_m = bs(phi, time, event)
    ipa_0 = (1 / 4) * np.trapz(1 - (bs_m / bs_n), timepoints, axis=0)

    ipa_1 = IPA(timepoints, integrate=True)(phi, time, event)

    assert all(ipa_0 == ipa_1)
