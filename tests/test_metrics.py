import torch
from hypothesis import given
from hypothesis import strategies as st

from discotime.metrics import BrierScore
from discotime.utils import AalenJohansen, KaplanMeier


@given(
    tau=st.lists(st.floats(min_value=0, max_value=3.1999), min_size=1),
)
def test_brier_score_1(tau, survival_data_2):
    """Before any event occurs (t=0.7) the following is true:

    - An estimate of 100% gives a brier score of 0
    - An estimate of 0% gives a brier score of 1
    - An estimate of 50% gives a brier score of 0.25

    """
    time, event = survival_data_2
    brier_score = BrierScore(survival_train=(time, event))

    all_ones = torch.ones((len(time), len(tau), 2))
    all_zeros = torch.zeros_like(all_ones)
    all_half = torch.full_like(all_ones, 0.5)

    brier_score(all_ones, tau, survival_data_2)
    assert torch.all(brier_score(all_ones, tau, (time, event)) == 0)
    assert torch.all(brier_score(all_zeros, tau, (time, event)) == 1)
    assert torch.all(brier_score(all_half, tau, (time, event)) == 0.25)


def brier_score_nested(estimates, timepoints, survival_test):
    """Implementation of the Brier score using nested for loops.

    Relatively inefficient, but the code is easier to read than the one relying
    on torch broadcasting rules. This implementation should give the same
    result as the vectorized one, otherwise a bug/error have been introduced
    somewhere - most likely in the "efficient" version.
    """
    time, event = map(torch.as_tensor, survival_test)
    S = torch.as_tensor(estimates)

    GTi = torch.clamp(KaplanMeier(time, event == 0)(time), 0.001)
    Gti = torch.clamp(KaplanMeier(time, event == 0)(timepoints), 0.001)

    out = torch.zeros_like(S)
    for i, (Ti, di) in enumerate(zip(time, event)):
        for j, t in enumerate(timepoints):
            for k in range(S.shape[-1]):
                surv = S[i][j][k]
                p1 = (surv**2 * ((Ti <= t) & (di == k + 1))) / GTi[i]
                p2 = ((1 - surv) ** 2 * ((Ti <= t) & (di != k + 1))) / GTi[i]
                p3 = ((1 - surv) ** 2 * (Ti > t)) / Gti[j]
                out[i, j, k] = p1 + p2 + p3

    return torch.mean(out, dim=0)


@given(
    tau=st.lists(st.floats(min_value=0, max_value=24.3), min_size=1),
    est=st.floats(min_value=0, max_value=1),
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
