import xarray as xr
import numpy as np


class FeatureEngineer:

    def add_temperature_features(self, ds):
        ds["temp_mean"] = (
            ds.tmax + ds.tmin
        ) / 2
        ds["temp_range"] = (
            ds.tmax - ds.tmin
        )
        return ds
    
    def add_rainfall_features(self, ds):
        ds["rain_7day"] = (ds.rainfall.rolling(time=7).mean())
        ds["rain_30day"] = (ds.rainfall.rolling(time=30).mean())
        return ds
    
    def add_lag_features(self, ds):
        ds["rain_lag1"] = ds.rainfall.shift(time=1)
        ds["rain_lag3"] = ds.rainfall.shift(time=3)
        ds["rain_lag7"] = ds.rainfall.shift(time=7)
        return ds
    
    def add_calendar_features(self, ds):
        ds["month"] = ds.time.dt.month
        ds["dayofyear"] = ds.time.dt.dayofyear
        return ds
    
    def add_season(self, ds):
        month = ds.time.dt.month
        season = xr.where(month.isin([12,1,2]), 0, 
                    xr.where(month.isin([3,4,5]), 1, 
                    xr.where( month.isin([6,7,8]), 2, 3)))
        ds["season"] = season
        return ds
    
    def add_anomaly(self, ds):

        climatology = (ds.rainfall.groupby("time.dayofyear").mean(dim="time"))

        anomaly = (ds["rainfall"].groupby("time.dayofyear") - climatology)

        anomaly = anomaly.reset_coords(drop=True)

        ds["rain_anomaly"] = anomaly

        return ds
    
    def build(self, ds):
        """
        Build all engineered features for the climate dataset.
        """

        print("Adding temperature features...")
        ds = self.add_temperature_features(ds)

        print("Adding rainfall features...")
        ds = self.add_rainfall_features(ds)

        print("Adding lag features...")
        ds = self.add_lag_features(ds)

        print("Adding calendar features...")
        ds = self.add_calendar_features(ds)

        print("Adding seasonal features...")
        ds = self.add_season(ds)

        print("Adding rainfall anomaly...")
        ds = self.add_anomaly(ds)

        print("Feature engineering complete.")

        return ds