import pytest
import numpy as np


@pytest.fixture(scope="session")
def comprisk_testdata():
    """Example data from Chap. 9 in Kleinbaum & Klein (2005)."""

    # [[censored], [cause_1], [cause_2]]
    data = [
        [3.2, 7.6, 10, 11, 15, 24.4],
        [0.7, 3, 4.9, 6, 6, 6.9, 10, 10.8, 17.1, 20.3],
        [1.5, 2.8, 3.8, 4.7, 7, 10, 10, 11.2],
    ]
    time, event = zip(
        *[(v, i) for (i, times) in enumerate(data) for v in times]
    )
    time, event = map(np.asarray, (time, event))
    return (time, event)
