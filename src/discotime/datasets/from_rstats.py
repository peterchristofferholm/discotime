from pathlib import Path
from typing import Optional
from functools import cached_property

from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torch import Generator
import requests
import rdata
import numpy as np
import pandas as pd

import discotime
from discotime.datasets.utils import (
    LitSurvDataModule,
    DataConfig,
    default_fts_transformer,
    SurvDataset,
)
from discotime.datasets.utils import LabelDiscretizer


_ROOT_PATH = Path(discotime.__file__).parent
_DATA_PATH = _ROOT_PATH / "datasets" / "_data"


class Mgus2(LitSurvDataModule):
    """
    Natural history of 1341 sequential patients with monoclonal gammopathy of
    undetermined significance (MGUS).

    [1]: R. Kyle, T. Therneau, V. Rajkumar, J. Offord, D. Larson, M. Plevak,
        and L. J. Melton III, A long-terms study of prognosis in monoclonal
        gammopathy of undertermined significance. New Engl J Med, 346:564-569
        (2002).
    """

    n_features = 7
    n_risks = 2

    def __init__(
        self,
        data_config: DataConfig,
        data_dir: Path = _DATA_PATH,
        seed: int = 13411341,
    ) -> None:
        super().__init__()

        self.seed = seed
        self.rng = Generator().manual_seed(seed)
        self.mgus2_csv = data_dir / "mgus2.csv"

        self.fts_transformer = default_fts_transformer()
        self.lab_transformer = LabelDiscretizer(
            n_bins=data_config.n_time_bins,
            scheme=data_config.discretization_scheme,
        )

        self._config = data_config
        self._dset_fit: Optional[SurvDataset] = None
        self._dset_test: Optional[SurvDataset] = None

        self.save_hyperparameters(ignore=["data_dir"])

    def prepare_data(self) -> None:
        """Fetch the built-in mgus2 from the R survival package.

        The mgus2 data [1] is extracted from the rdata file and saved as a csv
        in the discotime installation directory. If the csv is already
        available, then the download logic is skipped.

        [1]: Therneau T (2023). A Package for Survival Analysis in R.
        """
        url = "https://github.com/therneau/survival/raw/649851/data/cancer.rda"

        if not self.mgus2_csv.is_file():
            # create data directory if needed
            self.mgus2_csv.parent.mkdir(parents=True, exist_ok=True)
            # download and parse rdata file
            r = requests.get(url)
            parsed = rdata.parser.parse_data(r.content)
            df = rdata.conversion.convert(parsed)["mgus2"]
            # slight reformatting
            x = df.loc[:, "age":"mspike"]  # type: ignore
            t = df[["futime", "ptime"]].min(axis=1)
            e = pd.concat((df["pstat"] * 2, df["death"]), axis=1).max(axis=1)
            y = pd.concat({"time": t, "event": e}, axis=1)
            # combine and save as csv
            pd.concat((x, y), axis=1).to_csv(self.mgus2_csv, index=False)

    @property
    def dset_fit(self) -> SurvDataset:
        if self._dset_fit is None:
            raise AttributeError(
                "`dset_fit` not initialized yet."
                "Maybe try calling `.setup()` first?"
            )
        return self._dset_fit

    @property
    def dset_test(self) -> SurvDataset:
        if self._dset_test is None:
            raise AttributeError(
                "`dset_fit` not initialized yet."
                "Maybe try calling `.setup()` first?"
            )
        return self._dset_test

    @cached_property
    def time_range(self):
        data = pd.read_csv(self.mgus2_csv, dtype={"event": np.int32})
        data = data[data.event != 0]
        time = data.groupby(["event"])["time"]
        return (time.min().max(), time.max().min())

    def setup(self, stage: Optional[str] = None) -> None:
        if self._dset_fit is None:
            data = pd.read_csv(self.mgus2_csv, dtype={"event": np.int32})
            features = data.loc[:, "age":"mspike"]  # type: ignore
            outcomes = data.loc[:, ["time", "event"]]

            X_fit, X_test, Y_fit, Y_test = train_test_split(
                features, outcomes, test_size=0.2, random_state=self.seed
            )

            # prepare transformers
            self.lab_transformer.fit(Y_fit.time.values, Y_fit.event.values)
            self.fts_transformer.fit(X_fit)

            # initialize the datasets
            self._dset_fit = SurvDataset(
                features=self.fts_transformer.transform(X_fit),
                event_time=Y_fit.time.values,
                event_status=Y_fit.event.values,
                discretizer=self.lab_transformer,
            )
            self._dset_test = SurvDataset(
                features=self.fts_transformer.transform(X_test),
                event_time=Y_test.time.values,
                event_status=Y_test.event.values,
                discretizer=self.lab_transformer,
            )

        if stage in {"fit", "validate", "debug"}:
            self.dset_train, self.dset_val = random_split(
                dataset=self.dset_fit,
                lengths=[0.8, 0.2],
                generator=self.rng,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dset_val, batch_size=self.batch_size, num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_test, batch_size=self.batch_size, num_workers=1
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dset_test, batch_size=self.batch_size, num_workers=1
        )
