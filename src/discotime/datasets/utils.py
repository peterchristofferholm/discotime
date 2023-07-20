from typing import Optional
from collections.abc import Iterable
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
import numpy.typing as npt

from torch.utils.data import Dataset
import lightning.pytorch as pl

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from discotime.utils import KaplanMeier
from discotime.utils.typing import Int, Num, LabelTransformer, SurvData

###############################################################################


class SurvDataset(Dataset):
    r"""Assemble a survival dataset for discrete-time survival analysis.

    A discrete time survival dataset :math:`\mathfrak{D}` is a set of :math:`n`
    tuples :math:`(t_{i}, \delta_{i}, \mathbf{x}_{i})` where :math:`(t_i = \min
    \{T_i, C_i\})` is the event time, :math:`\delta_{i} \in \{0, ..., m\}`
    is the event indicator (with :math:`(\delta_i = 0)` defined as censoring),
    and :math:`\mathbf{x}_{i} \in \mathbb{R}^d` is a :math:`d`-dimensional
    vector of time-independent predictors or covariates.

    Args:
        features: time-independent features.
        event_time: follow-up time (continuous).
        event_status: event indicator (0=censored, 1/2/...=competing risks).
        discretizer: discretizer that follows the :py:class:`LabelTransformer`
            protocol that convert continuous time/event tuples to their
            respective discretized versions. Typically this would be
            :py:class:`LabelDiscretizer` unless a custom discretization object is
            used.
    """

    def __init__(
        self,
        features: npt.ArrayLike,
        event_time: npt.ArrayLike,
        event_status: npt.ArrayLike,
        discretizer: LabelTransformer,
    ) -> None:
        super().__init__()

        ct = np.asanyarray(event_time, dtype=np.int32)
        ce = np.asanyarray(event_status, dtype=np.int32)
        dt, de = discretizer.transform(ct, ce)

        self.features = np.asanyarray(features, dtype=np.float32)
        if not self.features.shape[0] == len(ct) == len(ce):
            raise ValueError(
                "`features`, `event_time`, and `event_status`"
                "should all have the same number of rows."
            )

        self.event_time_cont = ct
        self.event_status_cont = ce
        self.event_time_disc = dt
        self.event_status_disc = de

    def __len__(self):
        return len(self.event_time_cont)

    def __getitem__(self, index) -> SurvData:
        return SurvData(
            features=self.features[index, ...],
            event_time_disc=self.event_time_disc[index],
            event_status_disc=self.event_status_disc[index],
            event_time_cont=self.event_time_cont[index],
            event_status_cont=self.event_status_cont[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


###############################################################################


@dataclass(kw_only=True)
class DataConfig:
    """Configuration class for data modules."""

    batch_size: int = 32
    """
    The batch size defines the number of samples that will be propagated
    through the network at each training step.
    """
    n_time_bins: int = 20
    """
    Specifies the size of the discretization grid.
    A default of around 20-30 usually works good.
    """
    discretization_scheme: str = "number"
    discretization_grid: Optional[list[float]] = None
    max_time: Optional[float] = None


class LitSurvDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self._dset_fit: Optional[SurvDataset] = None
        self._dset_test: Optional[SurvDataset] = None
        self._config: Optional[DataConfig] = None

    def __repr__(self) -> str:
        return (
            super().__repr__() + "\n    "
            "n_features: {}, n_risks: {}, n_time_bins: {}".format(
                self.n_features, self.n_risks, self.n_time_bins
            )
        )

    @property
    def config(self) -> DataConfig:
        if self._config is None:
            raise AttributeError("`self._config` needs to be defined.")
        return self._config

    @property
    def cuts(self) -> npt.NDArray[np.float_]:
        return self.lab_transformer.cuts

    @property
    def dset_fit(self) -> SurvDataset:
        if self._dset_fit is None:
            raise AttributeError("`self._dset_fit` has not been prepared.")
        return self._dset_fit

    @property
    def dset_test(self) -> SurvDataset:
        if self._dset_test is None:
            raise AttributeError("`self._dset_test` has not been prepared.")
        return self._dset_test

    @property
    def n_time_bins(self) -> int:
        return self.config.n_time_bins

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    @abstractmethod
    def time_range(self) -> tuple[float, float]:
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        ...

    @property
    @abstractmethod
    def n_risks(self) -> int:
        ...

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        ...

    @property
    def lab_transformer(self) -> LabelTransformer:
        return self._lab_transformer

    @lab_transformer.setter
    def lab_transformer(self, value: LabelTransformer) -> None:
        self._lab_transformer = value


###############################################################################


def default_fts_transformer():
    transformer = make_column_transformer(
        # apply one-hot encoding to str columns
        (
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include=object),
        ),
        # apply z-scaling and mean-imputation to numeric colums
        (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("impute", SimpleImputer(strategy="constant")),
                ]
            ),
            make_column_selector(dtype_include=np.number),
        ),
        remainder="passthrough",
    )
    return transformer


###############################################################################


def _discretize_labels(
    time: Iterable[Num],
    event: Iterable[Int],
    cuts: Iterable[Num],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Discretize continous time/event pairs using supplied cuts."""
    _time, _event, _cuts = map(np.asanyarray, (time, event, cuts))

    maxi = len(_cuts) - 1
    de = np.atleast_1d(np.copy(_event)).astype(np.int64)
    de[_time > _cuts[-1]] = 0  # censored

    _time = np.atleast_1d(np.clip(_time, a_min=0, a_max=_cuts[-1]))
    dt = (np.searchsorted(_cuts, _time, side="right") - 1).astype(np.int64)

    # give credit for surviving more than half of an interval
    where = (de == 0) & (dt != maxi)
    x1, x2 = _cuts[dt[where]], _cuts[dt[where] + 1]
    dt[where] += (_time[where] - x1) > (x2 - _time[where])

    dt[dt == maxi] -= 1
    return (dt, de)


def _cuts_interval(n_bins: Int, max_time: Num) -> npt.NDArray[np.float64]:
    """Create cuts that define n groups with equal range.

    Example:
        >>> _cuts_interval(5, 10)
        array([ 0.,  2.,  4.,  6.,  8., 10.])
    """

    return np.linspace(0, max_time, n_bins + 1, dtype=np.float64)


def _cuts_number(
    n_bins: Int,
    max_time: Num,
    time: Iterable[Num],
    event: Iterable[Int],
) -> npt.NDArray[np.float64]:
    """Makes n groups with approx. equal number of observations."""
    _time, _event = map(np.asarray, (time, event))
    km = KaplanMeier(_time, (_event != 0).astype(np.int_))
    qs = np.linspace(1, km(max_time).item(), n_bins + 1)
    return np.asarray(km.percentile(qs), dtype=np.float64)


class LabelDiscretizer:
    """Discretize continous time/event pairs.

    The class can either learn a discretization grid from the training data
    using one of the built-in discretization schemes, or the user can supply an
    iterable with cut points.

    Implementation heavily inspired by pycox.preprocessing.label_tranform [1].

    [1]: Kvamme, Håvard, Ørnulf Borgan, and Ida Scheel. "Time-to-event
    prediction with neural networks and Cox regression." arXiv preprint
    arXiv:1907.00825 (2019).
    """

    def __init__(
        self,
        scheme: Optional[str] = None,
        n_bins: Optional[int] = None,
        *,
        cut_points: Optional[Iterable[Num]] = None,
        max_time: Optional[Num] = None,
    ) -> None:
        """Initialize LabTransDiscrete

        Available discretization schemes are:

        * interval: makes n groups with equal range
        * number: makes n groups with approx. equal number of observations
        * manual: supplying an iterable of cutpoints triggers the class to use
            those numbers to define the intervals instead.

        Args:
            n_bins: number of discretization bins to obtain.
            scheme: discretization scheme used to obtain cuts.
            cut_points: an iterable of pre-calculated cuts. Should only be
                used for manual specification of cut points. Default is [].
            max_time: truncate follow-up time. Default is None.
        """

        self._cuts = np.array([], np.float64)
        self._max_time = max_time

        if scheme in {"interval", "number"}:
            if isinstance(n_bins, int):
                self.n_bins = n_bins
                self.scheme = scheme
            else:
                raise ValueError("`n_bins` is not specified.")
        elif cut_points is not None:
            self._cuts = np.asarray(cut_points, np.float64)
            self.n_bins = len(self._cuts) - 1
            self.scheme = "manual"
        else:
            raise Exception(f"{self.__class__} could not be instantiated.")

    @property
    def max_time(self) -> Num:
        if self._max_time is None:
            raise AttributeError(
                "`max_time` not defined yet. Try calling `.fit()` first?"
            )
        return self._max_time

    @property
    def cuts(self) -> npt.NDArray[np.float64]:
        if not self._cuts.size:
            raise AttributeError(
                "`cuts` not defined yet. Try calling `.fit()` first?"
            )
        return self._cuts

    def fit(self, time: Iterable[Num], event: Iterable[Int]) -> None:
        self._max_time = self._max_time or max(time)

        match self.scheme:
            case "interval":
                self._cuts = _cuts_interval(self.n_bins, self.max_time)
            case "number":
                self._cuts = _cuts_number(
                    self.n_bins, self.max_time, time, event
                )
            case "manual":
                warnings.warn(
                    "ignoring fit(), scheme is set to 'manual'", UserWarning
                )

        if not np.unique(self.cuts).size == (self.n_bins + 1):
            raise ValueError(
                "Not all values in `self.cuts` are unique."
                f"Decrease the size of `n_bins` (currently {self.n_bins})"
            )

    def fit_transform(
        self, time: Iterable[Num], event: Iterable[Int]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        self.fit(time, event)
        return self.transform(time, event)

    def transform(
        self, time: Iterable[Num], event: Iterable[Int]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        return _discretize_labels(time, event, self.cuts)
