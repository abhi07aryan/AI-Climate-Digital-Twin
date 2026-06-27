from pathlib import Path
import calendar
import re

import numpy as np
import pandas as pd
import xarray as xr

from climate_twin.logger import logger


class IMDBaseLoader:
    """
    Generic loader for IMD gridded binary datasets.

    Child classes must define:

        VARIABLE_NAME
        N_LAT
        N_LON
        LAT_START
        LON_START
        RESOLUTION
        DTYPE
    """

    VARIABLE_NAME = None

    N_LAT = None
    N_LON = None

    LAT_START = None
    LON_START = None

    RESOLUTION = None

    DTYPE = np.float32

    def __init__(self):

        self._validate()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate(self):

        required = [

            "VARIABLE_NAME",

            "N_LAT",
            "N_LON",

            "LAT_START",
            "LON_START",

            "RESOLUTION"

        ]

        for attr in required:

            if getattr(self, attr) is None:

                raise ValueError(
                    f"{attr} must be defined "
                    f"in {self.__class__.__name__}"
                )

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def _extract_year(self, filepath: Path):

        match = re.search(
            r"(19|20)\d{2}",
            filepath.stem
        )

        if match is None:

            raise ValueError(
                f"Cannot extract year from {filepath.name}"
            )

        return int(match.group())

    def _coordinates(self):

        lat = (

            self.LAT_START

            +

            np.arange(self.N_LAT)

            *

            self.RESOLUTION

        )

        lon = (

            self.LON_START

            +

            np.arange(self.N_LON)

            *

            self.RESOLUTION

        )

        return lat, lon

    # ---------------------------------------------------------
    # Dataset specific hook
    # ---------------------------------------------------------

    def handle_missing(self, data):

        """
        Override in child class.
        """

        return data

    # ---------------------------------------------------------
    # Load one year
    # ---------------------------------------------------------

    def load_year(self, filepath):

        filepath = Path(filepath)

        if not filepath.exists():

            raise FileNotFoundError(filepath)

        logger.info(
            f"Loading {filepath.name}"
        )

        data = np.fromfile(

            filepath,

            dtype=self.DTYPE

        )

        year = self._extract_year(filepath)

        days = (

            366

            if calendar.isleap(year)

            else 365

        )

        expected = (

            days

            *

            self.N_LAT

            *

            self.N_LON

        )

        if data.size != expected:

            raise ValueError(

                f"Expected {expected:,} values "

                f"found {data.size:,}"

            )

        data = data.reshape(

            days,

            self.N_LAT,

            self.N_LON

        )

        data = data.astype(np.float32)

        data = self.handle_missing(data)

        lat, lon = self._coordinates()

        time = pd.date_range(

            start=f"{year}-01-01",

            periods=days,

            freq="D"

        )

        ds = xr.Dataset(

            data_vars={

                self.VARIABLE_NAME: (

                    ("time", "lat", "lon"),

                    data

                )

            },

            coords={

                "time": time,

                "lat": lat,

                "lon": lon

            },

            attrs={

                "source": "IMD",

                "resolution": self.RESOLUTION,

                "year": year

            }

        )

        logger.info(

            f"{self.VARIABLE_NAME}: "

            f"{ds[self.VARIABLE_NAME].shape}"

        )

        return ds

    # ---------------------------------------------------------
    # Load multiple years
    # ---------------------------------------------------------

    def load_all_years(self, folder):

        folder = Path(folder)

        files = sorted(

            folder.glob("*.GRD")

        )

        if not files:

            files = sorted(

                folder.glob("*.grd")

            )

        if not files:

            raise FileNotFoundError(

                f"No GRD files in {folder}"

            )

        logger.info(

            f"Found {len(files)} files."

        )

        datasets = [

            self.load_year(file)

            for file in files

        ]

        ds = xr.concat(

            datasets,

            dim="time"

        )

        logger.info(

            f"Final Dataset Shape: "

            f"{ds[self.VARIABLE_NAME].shape}"

        )

        return ds

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(self, dataset, path):

        path = Path(path)

        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        dataset.to_netcdf(path)

        logger.info(

            f"Saved dataset -> {path}"

        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self, dataset):

        var = dataset[self.VARIABLE_NAME]

        logger.info("=" * 60)

        logger.info(var)

        logger.info(
            f"Min      : {np.nanmin(var):.3f}"
        )

        logger.info(
            f"Max      : {np.nanmax(var):.3f}"
        )

        logger.info(
            f"Mean     : {np.nanmean(var):.3f}"
        )

        logger.info(
            f"NaNs     : {np.isnan(var).sum():,}"
        )

        logger.info("=" * 60)