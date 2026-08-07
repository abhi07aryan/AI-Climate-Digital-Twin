from pathlib import Path

import numpy as np
import xarray as xr

from climate_twin.logger import logger


class IMDBaseLoader:
    """
    Generic loader for IMD NetCDF datasets.
    """

    VARIABLE_NAME = None

    def __init__(self):
        if self.VARIABLE_NAME is None:
            raise ValueError(
                f"VARIABLE_NAME must be defined in {self.__class__.__name__}"
            )

    # ---------------------------------------------------------
    # Dataset specific hook
    # ---------------------------------------------------------

    def handle_missing(self, data):
        """
        Override in child class.
        """
        return data

    # ---------------------------------------------------------
    # Load one NetCDF file
    # ---------------------------------------------------------

    def load_year(self, filepath):

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(filepath)

        logger.info(f"Loading {filepath.name}")

        ds = xr.open_dataset(filepath)

        if self.VARIABLE_NAME not in ds:
            raise KeyError(
                f"{self.VARIABLE_NAME} not found in {filepath.name}"
            )

        ds[self.VARIABLE_NAME] = self.handle_missing(
            ds[self.VARIABLE_NAME]
        )

        logger.info(
            f"{self.VARIABLE_NAME}: {ds[self.VARIABLE_NAME].shape}"
        )

        return ds

    # ---------------------------------------------------------
    # Load multiple years
    # ---------------------------------------------------------

    def load_all_years(self, folder):

        folder = Path(folder)

        files = sorted(folder.glob("*.nc"))

        if not files:
            raise FileNotFoundError(
                f"No NetCDF files found in {folder}"
            )

        logger.info(f"Found {len(files)} files.")

        datasets = []

        for file in files:
            datasets.append(self.load_year(file))

        ds = xr.concat(datasets, dim="time")

        logger.info(
            f"Final Dataset Shape: {ds[self.VARIABLE_NAME].shape}"
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

        logger.info(f"Saved dataset -> {path}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self, dataset):

        var = dataset[self.VARIABLE_NAME]

        logger.info("=" * 60)
        logger.info(var)
        logger.info(f"Min      : {float(var.min(skipna=True)):.3f}")
        logger.info(f"Max      : {float(var.max(skipna=True)):.3f}")
        logger.info(f"Mean     : {float(var.mean(skipna=True)):.3f}")
        logger.info(f"NaNs     : {int(var.isnull().sum())}")
        logger.info("=" * 60)