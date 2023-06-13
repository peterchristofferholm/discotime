from typing import Optional
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch
from einops import repeat

Int = int | np.int_
Num = Int | float | np.float_


def _tabulate(a, v, side="left"):
    """Count interval occurences.

    As an example, if side="left" and a=[1, 2, 3], then intervals are
    (-inf, 1), [1, 2), [2, 3) and [3, inf)

    Example:
        >>> a, v = np.array([1, 2, 3]), np.array([0.5, 1, 2.5, 4])
        >>> _tabulate(a, v)
        array([2, 0, 1, 1])
    """
    return np.bincount(np.searchsorted(a, v, side=side))


class KaplanMeier:
    """Simple implementation of the Kaplan-Meier estimator.

    Args:
        time: event times.
        event: event indicator (0/1) where 0 is censoring.

    Example:
        >>> km = KaplanMeier(time=[0, 1.5, 1.3, 3], event=[0, 1, 0, 0])
        >>> km(0)
        array([1.])
        >>> km([0, 1.0, 1.1, 1.5])
        array([1. , 1. , 1. , 0.5])
    """

    def __init__(self, time: Iterable[Num], event: Iterable[Int]) -> None:
        _time, _event = map(np.asarray, (time, event))
        order = np.argsort(_time)
        _time, _event = _time[order], _event[order]

        # observed event times
        tj = np.unique(np.pad(_time[_event == 1], (1, 0)))

        def _pad(a):
            return np.pad(a, (0, tj.size - a.size))

        mj = _pad(_tabulate(tj, _time[_event == 1], side="right")[1:])
        qj = _pad(_tabulate(tj, _time[_event == 0], side="right")[1:])

        nj = np.roll(_time.size - np.cumsum(mj + qj), 1)
        nj[0] = _time.size

        self._sj = np.cumprod((nj - mj) / nj)
        self._nj, self._tj, self._mj = nj, tj, mj

    def __call__(self, time: Num | Iterable[Num]) -> npt.NDArray[np.float_]:
        """Obtain Kaplan-Meier estimates for each timepoint.

        Args:
            time (num | seq[num]): `t`'s for which `km(t)` will be returned.
        """
        _time = np.atleast_1d(np.asanyarray(time))
        return self._sj[np.searchsorted(self._tj, _time, side="right") - 1]

    def percentile(
        self, p: Iterable[Num], dtype=np.float64
    ) -> npt.NDArray[np.float_]:
        """Obtain approximate timepoint t such that P(t) = p.

        The stepwise Kaplan-Meier estimator is piecewise linearly interpolated
        such that unique timepoints can be obtained.
        """
        p = np.atleast_1d(np.asanyarray(p, dtype=dtype))
        if not np.all((0 <= p) & (p <= 1)):
            raise ValueError(
                "p is a probability and should be between 0 and 1"
            )
        return np.interp(1 - p, 1 - self._sj, self._tj, left=0)


class AalenJohansen:
    """Obtain cumulative incidence curves with the Aalen-Johansen method.

    Args:
        time: event times
        event: event indicator (0/1/../c) with 0=censored
        n_causes: how many causes should be included? If None (the default)
            then all observed causes are include.
    """

    def __init__(
        self,
        time: npt.ArrayLike,
        event: npt.ArrayLike,
        n_causes: Optional[Int] = None,
    ) -> None:
        time, event = map(np.asarray, (time, event))

        if time.ndim != 1:
            raise ValueError(f"`time` is a {time.ndim}D array.")

        if event.ndim != 1:
            raise ValueError(f"`event` is a {event.ndim}D array.")

        order = np.argsort(time)
        time, event = time[order], event[order]

        # find unique event times and add t=0 if it isn't present
        tj = np.unique(np.pad(time, (1, 0)))

        # number at risk at the start of each interval
        nj = time.size - np.cumsum(_tabulate(tj, time, side="right"))
        nj = nj[:-1]  # drop last

        def _pad(a):
            """Add zeros to end of array to ensure it has same size as `tj`"""
            assert (tj.size - a.size) >= 0
            return np.pad(a, (0, tj.size - a.size), constant_values=0)

        # number lost in each interval to any cause
        mj = _pad(_tabulate(tj, time[event != 0]))

        # lagged survival which starts with P(0) = 1
        sj = np.cumprod((nj - mj) / nj)[:-1]
        sj = np.pad(sj, (1, 0), constant_values=1)

        # cause-specific incidences
        n_risks = n_causes if n_causes else np.max(event)
        ci = np.zeros((tj.size, n_risks))
        for e in range(1, n_risks + 1):
            mcj = _pad(_tabulate(tj, time[event == e]))
            ci[:, e - 1] = np.cumsum(sj * (mcj / nj))

        self._tj = tj
        self._sj = sj
        self._ci = ci

    def __call__(self, timepoints: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return cause-specific cumulative incidence at given timepoints."""
        tau = np.asanyarray(timepoints).reshape(-1)
        idx = np.searchsorted(self._tj, tau, side="right")
        idx = np.clip(idx - 1, a_min=0, a_max=(self._tj.size - 1))
        return self._ci[idx]


def interpolate2d(x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor):
    """Perform stepwise linear interpolation of a discrete function.

    _xp_ and _yp_ are tensors of values used to approximate f: y = f(x). This
    functions uses interpolation to find the value of new points x.

    Args:
        x (:obj:`torch.Tensor`): an 1D tensor real values.
        xp (:obj:`torch.Tensor`): an 1D tensor of real values.
        yp (:obj:`torch.Tensor`): an ND tensor of real values. The length of yp
            along the second axis (dim=1) must have the same length as xp.
    """
    x, xp = x.to(yp), xp.to(yp)  # move to same device
    m = torch.diff(yp, dim=1) / repeat(torch.diff(xp, dim=0), "t -> 1 t 1")
    b = yp[:, :-1, :] - torch.einsum("btr,t->btr", m, xp[:-1])
    idx = torch.clamp_max(torch.searchsorted(xp, x), m.shape[1] - 1)
    return m[:, idx, :] * repeat(x, "t -> 1 t 1") + b[:, idx, :]
