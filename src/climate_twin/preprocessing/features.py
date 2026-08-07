import xarray as xr
import numpy as np

class FeatureEngineer:

    def add_temperature_features(self, ds):
        
        ds["temp_mean"] = (ds.tmax + ds.tmin) / 2

        ds["temp_range"] = ds.tmax - ds.tmin

        return ds

    def add_rainfall_features(self, ds):

        ds["rain_7day"] = ds.rainfall.rolling(
            time=7,
            min_periods=1
        ).mean()

        ds["rain_30day"] = ds.rainfall.rolling(
            time=30,
            min_periods=1
        ).mean()

        return ds

    def add_calendar_features(self, ds):

        ds["month"] = ds.time.dt.month

        ds["dayofyear"] = ds.time.dt.dayofyear

        return ds

    def add_anomaly(self, ds):

        climatology = ds.rainfall.groupby(
            "time.dayofyear"
        ).mean(dim="time")

        ds["rain_anomaly"] = (
            ds.rainfall.groupby("time.dayofyear")
            - climatology
        ).reset_coords(drop=True)

        return ds

    def build(self, ds):
        """
        Build all engineered features.
        """
        print("Adding temperature features...")
        ds = self.add_temperature_features(ds)

        print("Adding rainfall features...")
        ds = self.add_rainfall_features(ds)

        print("Adding calendar features...")
        ds = self.add_calendar_features(ds)

        print("Adding rainfall anomaly...")
        ds = self.add_anomaly(ds)

        print("Feature engineering complete.")

        return ds