from typing import Optional
import numpy as np
import numpy.typing as npt
from discotime.metrics import BrierScore


class IPA:
    """Index of prediction accuracy (IPA) for right-censored survival models.

    In the context of survival analysis, the IPA is also known as the Brier
    skill score (BSS).
    """

    def __init__(
        self,
        timepoints: npt.NDArray[np.float_],
        causes: Optional[list[int]] = None,
        integrate: bool = True,
    ) -> None:
        """
        Args:
            timepoints: sequence of `t` timepoints to evaluate.
            causes: causes for which the brier score is calculated.
            integrate: should the time-dependent IPA be integrated?
                (default: True)
        """
        self.timepoints = np.asanyarray(timepoints)
        self.brier_score = BrierScore(self.timepoints, causes)
        self.integrate = integrate

    def __call__(
        self,
        estimates: npt.NDArray[np.floating],
        time: npt.NDArray[np.floating],
        event: npt.NDArray[np.integer],
    ) -> np.float_:
        """Calculate the index of prediction accuracy.

        Args:
            estimates: array[batch, timepoints, cause]
                model estimates of cause probabilities.
            time: observed follow-up.
            event: event indicator at time with 0 as censoring.

        If there are multiple timepoints to consider, then the IPA is
        numerically integrated using the trapezoidal method.
        """

        bs = self.brier_score(estimates, time, event)

        with np.errstate(divide="ignore"):
            ipa = 1 - (bs.model / bs.null)
            ipa = np.nan_to_num(ipa, neginf=0, posinf=0)

        if len(self.timepoints) > 1 and self.integrate:
            scaling = 1 / (max(self.timepoints) - min(self.timepoints))
            ipa = np.trapz(ipa, self.timepoints, axis=0) * scaling

        return ipa
