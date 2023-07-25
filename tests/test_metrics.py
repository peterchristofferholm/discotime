from itertools import pairwise, count, islice

import torch
import pytest
from hypothesis import given, example
from hypothesis import strategies as st

from discotime.metrics import BrierScore, BrierScoreScaled
from discotime.utils import AalenJohansen, IPCW


@given(
    tau=st.lists(st.floats(0, 0.699), min_size=1, max_size=100),
)
def test_brier_score_1(tau, survival_data_2):
    """Before any event occurs (t=0.7) the following is true:

    - An estimate of 100% gives a brier score of 1
    - An estimate of 0% gives a brier score of 0
    - An estimate of 50% gives a brier score of 0.25

    """
    time, event = survival_data_2
    brier_score = BrierScore(survival_train=(time, event))

    all_ones = torch.ones((len(time), len(tau), 2))
    all_zeros = torch.zeros_like(all_ones)
    all_half = torch.full_like(all_ones, 0.5)

    assert torch.all(brier_score(all_ones, tau, (time, event)) == 1)
    assert torch.all(brier_score(all_zeros, tau, (time, event)) == 0)
    assert torch.all(brier_score(all_half, tau, (time, event)) == 0.25)


def brier_score_nested(estimates, timepoints, survival_test):
    """Implementation of the Brier score using nested for loops.

    Relatively inefficient, but the code is easier to read than the one relying
    on torch broadcasting rules. This implementation should give the same
    result as the vectorized one, otherwise a bug/error have been introduced
    somewhere - most likely in the "efficient" version.
    """
    futime, status = map(torch.as_tensor, survival_test)
    tau = torch.as_tensor(timepoints)
    St = torch.as_tensor(estimates)

    Gt = IPCW(futime, status)(tau, lag=0)
    GT = IPCW(futime, status)(futime, lag=1)

    out = torch.zeros_like(St)
    for i, (ti, s) in enumerate(zip(futime, status)):
        for j, t in enumerate(tau):
            for k in range(2):
                p = St[i][j][k]
                if ti <= t and s - 1 == k:
                    out[i, j, k] = (1 - p) ** 2 / GT[i]
                elif ti <= t and s != 0:
                    out[i, j, k] = p**2 / GT[i]
                elif ti > t:
                    out[i, j, k] = p**2 / Gt[j]

    return torch.mean(out, dim=0)


@given(
    tau=st.lists(st.floats(0, 24.3), min_size=1, max_size=100),
    est=st.floats(0, 1),
)
def test_brier_score_2(tau, est, survival_data_2):
    time, event = survival_data_2
    brier_score = BrierScore(survival_train=(time, event))
    phi = torch.full((len(time), len(tau), 2), est)

    # the discotime-packaged implementation
    brier_score_discotime = brier_score(
        estimates=phi,
        timepoints=tau,
        survival_test=(time, event),
    )

    assert brier_score_discotime.shape == phi.shape[1:]
    assert torch.all(
        (0 <= brier_score_discotime) & (brier_score_discotime <= 1)
    )

    # the stupid implementation relying on nested for-loops
    brier_score_naive = brier_score_nested(
        estimates=phi,
        timepoints=tau,
        survival_test=(time, event),
    )

    assert torch.all(brier_score_discotime == brier_score_naive)


@given(
    tau=st.lists(st.floats(0, 24.3), min_size=1, max_size=10),
    est=st.floats(0, 1),
)
def test_brier_score_3(tau, est, survival_data_2):
    """
    Guessing the incidence using the AalenJohansen estimator should be at least
    as good as any random constant guess.
    """
    time, event = survival_data_2
    brier_score = BrierScore(survival_train=(time, event))

    phi_test = torch.full((len(time), len(tau), 2), est)
    phi_null = AalenJohansen(time, event)(tau)
    phi_null = torch.as_tensor(phi_null).view(1, -1, 2).expand_as(phi_test)

    get_score = lambda x: brier_score(x, tau, (time, event))

    assert torch.all(get_score(phi_null).mean() <= get_score(phi_test).mean())


@pytest.mark.parametrize(
    ("cause", "tau", "expected"),
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
def test_brier_score_null(tau, cause, expected, survival_data_2):
    """Expected values obtained using the `riskRegression` R package.

    ```{r}
    cfit <- CSC(Hist(t, e) ~ 1, cause = _, data = data)
    Score(list(cfit), formula=Hist(t, e) ~ 1, data=data, times=c(...), cause=_)
    ```
    """
    futime, status = survival_data_2
    tau = sorted(tau)

    phi_null = AalenJohansen(futime, status)(tau)
    phi_null = (
        torch.as_tensor(phi_null)
        .view(1, -1, 2)
        .expand((len(futime), len(tau), 2))
    )

    brier_score = BrierScore(survival_train=(futime, status))
    result = brier_score(phi_null, tau, (futime, status))[..., cause - 1]
    assert result == pytest.approx(expected, abs=1e-3)


@given(
    tau=st.lists(st.floats(0, 50), min_size=2, unique=True),
    est=st.floats(0, 1),
    integrate=st.booleans(),
)
@example(
    tau=[0.0, 3.503246160812043e-46],
    est=0.0,
    integrate=True,
)
def test_brier_score_scaled_1(tau, est, integrate, survival_data_2):
    tau = sorted(tau)
    time, event = survival_data_2
    brier_score_scaled = BrierScoreScaled(
        survival_train=(time, event), eval_grid=tau, integrate=integrate
    )
    phi = torch.full((len(time), len(tau), 2), est)
    bss = brier_score_scaled(phi, survival_data_2)

    expected_ndim = 1 if integrate else 2
    assert bss.ndim == expected_ndim

    expected_shape = (2,) if integrate else (len(tau), 2)
    assert bss.shape == expected_shape

    assert torch.all(bss <= 1)


def test_brier_score_scaled_2(default_mgus2_model):
    model = default_mgus2_model
    data = [*map(torch.as_tensor, model.datamodule.dset_fit[:])]

    tau = torch.linspace(10, 365, 50)
    bss = BrierScoreScaled((data[3], data[4]), tau, True)

    def training_loop():
        optim = model.configure_optimizers()

        for i in count():
            model.train()
            for batch_idx, batch in enumerate(
                model.datamodule.train_dataloader()
            ):
                optim.zero_grad()
                loss = model.training_step(batch, batch_idx)
                model.backward(loss)
                optim.step()

            model.eval()
            est = model._predict_estimates(data[0], tau)

            if i % 5 == 0:
                yield bss(est, (data[3], data[4]))

    assert all(
        torch.all(a <= (b + 0.01))
        for (a, b) in pairwise(islice(training_loop(), 20))
    )
