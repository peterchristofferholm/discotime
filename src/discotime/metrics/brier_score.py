from typing import Optional, TypeVar
from collections import namedtuple

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from discotime.utils.estimators import AalenJohansen

T_co = TypeVar("T_co", bound=np.generic, covariant=True)

Int = int | np.int_
Num = Int | float | np.float_


def _loo_idx(m: int) -> npt.NDArray[np.int64]:
    """Create indices for leave-one-out resampling.

    Example:
        >>> _loo_idx(4)
        array([[1, 2, 3],
               [0, 2, 3],
               [0, 1, 3],
               [0, 1, 2]])
    """
    return np.arange(1, m, dtype=np.int64) - np.tri(m, m - 1, k=-1, dtype=bool)


def _cic_pv(
    time: npt.NDArray[np.float_],
    event: npt.NDArray[np.int_],
    tau: npt.ArrayLike,
) -> NDArray[np.float64]:
    """Obtain jackknife pseudovalues from the marginal Aalen-Johansen estimate.

    Example:
        >>> rng = np.random.default_rng()
        >>> time, event = rng.weibull(5., size=100), rng.integers(3, size=100)
        >>> aj = _cic_pv(time, event, tau=[0.5, 3, 4])
        >>> aj.shape
        (100, 3, 2)
    """
    N = time.size
    tau = np.asarray(tau, np.float_)

    n_causes = np.max(event)
    cumulative_incidence = AalenJohansen(time, event, n_causes)(tau)
    cumulative_incidence_loo = np.array(
        [AalenJohansen(time[i], event[i], n_causes)(tau) for i in _loo_idx(N)]
    )
    return (N * cumulative_incidence) - ((N - 1) * cumulative_incidence_loo)


class BrierScore:
    """Brier score of survival model with right-censored data.

    In the case of right-censored data, possibly unknown time-dependent event
    status can be replaced with jackknife pseudovalues from the marginal
    Aalen-Johansen estimates [1].

    Args:
        timepoints: sequence of `t` timepoints to evaluate.
        causes: causes for which the brier score is calculated.

    Refs:
        [1] Cortese, Giuliana, Thomas A. Gerds, and Per K. Andersen. "Comparing
        predictions among competing risks models with time‐dependent
        covariates." Statistics in medicine 32.18 (2013): 3089-3101.
    """

    Results = namedtuple("Results", ["null", "model"])

    def __init__(
        self, timepoints: npt.ArrayLike, causes: Optional[list[int]] = None
    ) -> None:
        if causes is not None and 0 in causes:
            raise ValueError("cause 0 is censoring and cannot be included.")
        self.timepoints = np.asanyarray(timepoints)
        self.causes = causes

    @staticmethod
    def _brier_score(label, estimate) -> NDArray[T_co]:
        estimate = np.asanyarray(estimate)
        score = label * (1 - (2 * estimate)) + np.square(estimate)
        return np.atleast_1d(np.mean(score, axis=0))

    def __call__(
        self,
        estimates: Optional[NDArray[T_co]],
        time: npt.ArrayLike,
        event: npt.ArrayLike,
    ) -> Results:
        """Calculate Brier score of survival model.

        Also includes the Brier score of a null-model based on the cumulative
        incidence curve obtained with the Aalen-Johansen estimator.

        Args:
            estimates: an array with shape (`m`, `t`, `e`), where `m` is the
                batch size, `t` is the number of time bins, and `e` is number
                of competing causes/risks.
            time: survival time.
            event: event indicator with event=0 defined censoring.
        """
        time, event = np.asanyarray(time), np.asanyarray(event)

        if not np.issubdtype(event.dtype, np.integer):
            raise TypeError("`event` can only be of integer type.")

        # check dimensionality of input arrays
        if time.ndim != 1:
            raise ValueError(f"`time` is a {time.ndim}D array.")
        if event.ndim != 1:
            raise ValueError(f"`event` is a {event.ndim}D array.")

        # use all available causes if none is specified
        if self.causes is None:
            causes = np.trim_zeros(np.unique(event))
        else:
            causes = np.asanyarray(self.causes)

        # obtain jack-knife pseudovalues
        cic_pv = _cic_pv(time, event, self.timepoints)[..., causes - 1]

        # the null model is the cic from the AalenJohansen estimator
        p_null = AalenJohansen(time, event)(self.timepoints)[..., causes - 1]
        bs_null: npt.NDArray[np.float32] = self._brier_score(cic_pv, p_null)

        # calculate brier score on the model estimates
        if estimates is not None:
            bs_model = self._brier_score(cic_pv, estimates[..., causes - 1])  # type: ignore
        else:
            bs_model = None

        return self.Results(null=bs_null, model=bs_model)
