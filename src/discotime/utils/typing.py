from typing import Protocol, Any, NamedTuple
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

Int = int | np.int_
Num = Int | float | np.float_


class Transformer(Protocol):
    def fit(self, *args, **kwargs) -> None:
        ...

    def transform(self, *args, **kwargs) -> Any:
        ...

    def fit_transform(self, *args, **kwargs) -> Any:
        ...


class LabelTransformer(Transformer, Protocol):
    """LabelTransformer"""

    def fit(self, time: Iterable[Num], event: Iterable[Int]) -> None:
        ...

    def transform(
        self,
        time: Iterable[Num],
        event: Iterable[Int],
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        ...

    def fit_transform(
        self,
        time: Iterable[Num],
        event: Iterable[Int],
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        ...

    @property
    def max_time(self) -> Num:
        ...

    @property
    def cuts(self) -> npt.NDArray[np.floating]:
        ...


class SurvData(NamedTuple):
    features: npt.NDArray[np.float_]
    event_time_disc: npt.NDArray[np.int_]
    event_status_disc: npt.NDArray[np.int_]
    event_time_cont: npt.NDArray[np.float_]
    event_status_cont: npt.NDArray[np.int_]
