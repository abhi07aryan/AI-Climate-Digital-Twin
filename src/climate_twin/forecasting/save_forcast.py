from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class ForecastWriter:

    def __init__(self):
        pass

    def save(
        self,
        forecast,
        lat,
        lon,
        start_date,
        filepath="results/forecast.nc"
    ):
        """
        Parameters
        ----------
        forecast : list or ndarray
            Shape:
            (days, height, width)

        lat : ndarray

        lon : ndarray

        start_date : str
            Example:
            "2025-01-01"
        """

        forecast = np.asarray(forecast)

        dates = pd.date_range(
            start=start_date,
            periods=forecast.shape[0],
            freq="D"
        )

        ds = xr.Dataset(

            data_vars=dict(

                rainfall=(

                    ("time", "lat", "lon"),

                    forecast

                )

            ),

            coords=dict(

                time=dates,

                lat=lat,

                lon=lon

            )

        )

        ds.rainfall.attrs["units"] = "mm"

        ds.rainfall.attrs["long_name"] = \
            "Forecast Rainfall"

        filepath = Path(filepath)

        filepath.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        ds.to_netcdf(filepath)

        print(f"Forecast saved to {filepath}")

        return ds