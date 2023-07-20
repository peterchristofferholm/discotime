from typing import TypeVar
from itertools import pairwise

import torch
from discotime.utils import AalenJohansen, IPCW

TensorLike = TypeVar("TensorLike")


class BrierScore:
    """Brier score of survival model with right-censored data.

    Args:
        survival_train: sequence of `t` timepoints to evaluate.
    """

    @torch.no_grad()
    def __init__(self, survival_train: tuple[TensorLike, TensorLike]) -> None:
        futime, status = survival_train
        futime = torch.as_tensor(futime).squeeze()
        status = torch.as_tensor(status).squeeze()

        if futime.ndim > 1 or status.ndim > 1:
            raise ValueError("`futime` and `status` should be 1D tensors")

        self.ipcw = IPCW(futime, status)

    @torch.no_grad()
    def __call__(
        self,
        estimates: TensorLike,
        timepoints: TensorLike,
        survival_test: tuple[TensorLike, TensorLike],
    ) -> torch.Tensor:
        """Calculate Brier score of survival model.

        Args:
            estimates: an array with shape (`m`, `t`, `e`), where `m` is the
                batch size, `t` is the number of time bins, and `e` is number
                of competing causes/risks.
            timepoints: array of timepoints with size `t` at which the
                values in `estimates` are obtained.
            survival_test: tuple of observed time/event data.
        """

        St = torch.as_tensor(estimates)
        n_obs, n_time, n_risks = St.shape

        tau = torch.as_tensor(timepoints)
        if len(tau) != n_time:
            raise ValueError(
                "size of `timepoints` do not match dim 1 of estimates."
            )

        futime, status = survival_test
        futime = torch.as_tensor(futime).squeeze()
        status = torch.as_tensor(status).squeeze()

        if futime.ndim > 1 or status.ndim > 1:
            raise ValueError("`futime` and `status` should be 1D tensors")

        if len(futime) != n_obs:
            raise ValueError("length of `futime` is not equal to batch size.")

        if len(status) != n_obs:
            raise ValueError("length of `status` is not equal to batch size.")

        # has individual experienced event of interest yet?
        I1 = torch.logical_and(
            futime.view(-1, 1, 1) <= tau.view(1, -1, 1),
            status.view(-1, 1, 1) == torch.arange(1, n_risks + 1),
        )
        # ... did something else happen?
        # fmt: off
        I2 = (
            (futime.view(-1, 1, 1) <= tau.view(1, -1, 1))
            .logical_and(status.view(-1, 1, 1) != torch.zeros_like(I1))
            .logical_and(status.view(-1, 1, 1) != torch.arange(1, n_risks + 1))
        )
        # fmt: on
        # ... or is individual still in the risk set?
        I3 = futime.view(-1, 1, 1) > tau.view(1, -1, 1)

        # The IPCW is clamped such that one individual is not "dominating" the
        # score. see Kvamme & Borgan. "The Brier Score under Administrative
        # Censoring: Problems and a Solution."
        Gt = torch.clamp(self.ipcw(tau, lag=0).reshape(1, -1, 1), min=1e-4)
        GTi = torch.clamp(self.ipcw(futime, lag=1).reshape(-1, 1, 1), min=1e-4)

        results = I1 * (1 - St) ** 2 / GTi
        results += I2 * (St**2) / GTi
        results += I3 * (St**2) / Gt

        return torch.mean(results, dim=0)


class BrierScoreScaled:
    """Index of prediction accuracy (IPA) for right-censored survival models.

    In the context of survival analysis, the IPA is also known as the Brier
    skill score (BSS).
    """

    @torch.no_grad()
    def __init__(
        self,
        survival_train: tuple[TensorLike, TensorLike],
        eval_grid: TensorLike,
        integrate: bool = False,
    ) -> None:
        """
        Args:
            timepoints: sequence of `t` timepoints to evaluate.
            causes: causes for which the brier score is calculated.
            integrate: should the time-dependent IPA be integrated?
                (default: True)
        """
        eval_grid = torch.as_tensor(eval_grid)

        if len({*eval_grid}) != len(eval_grid):
            raise ValueError("values in `timepoints` are not unique")

        if not all(a <= b for a, b in pairwise(eval_grid)):
            raise ValueError("values in `timepoints` are not sorted")

        if integrate and len(eval_grid) == 1:
            raise ValueError(
                "at least two values in `timepoints` are required when "
                "self.integrate = True"
            )

        self.brier_score = BrierScore(survival_train=survival_train)
        self.eval_grid = eval_grid
        self.integrate = integrate

    @torch.no_grad()
    def __call__(
        self,
        estimates: TensorLike,
        survival_test: tuple[TensorLike, TensorLike],
    ) -> torch.Tensor:
        """Calculate the index of prediction accuracy.

        Args:
            estimates: array[batch, timepoints, cause]
                model estimates of cause probabilities.
            time: observed follow-up.
            event: event indicator at time with 0 as censoring.

        If there are multiple timepoints to consider, then the IPA is
        numerically integrated using the trapezoidal method.
        """
        phi_test = torch.as_tensor(estimates)
        n_causes = phi_test.shape[-1]
        tau = self.eval_grid

        futime, status = survival_test
        futime = torch.as_tensor(futime).squeeze()
        status = torch.as_tensor(status).squeeze()

        cic = 1 - AalenJohansen(futime, status, n_causes)(tau)
        phi_null = cic.unsqueeze(0).expand_as(phi_test)

        bs_test = self.brier_score(phi_test, tau, survival_test)
        bs_null = self.brier_score(phi_null, tau, survival_test)

        if self.integrate:
            bs_null = torch.trapezoid(bs_null, tau.view(-1, 1), dim=0)
            bs_test = torch.trapezoid(bs_test, tau.view(-1, 1), dim=0)

        # avoid zero division
        bs_test[bs_null == 0] += 1e-10
        bs_null[bs_null == 0] += 1e-10

        return 1 - bs_test / bs_null
