import xarray as xr

from climate_twin.preprocessing.features import FeatureEngineer

# Load clipped datasets
rainfall = xr.open_dataset("data/interim/rainfall_up.nc")
tmax = xr.open_dataset("data/interim/tmax_up.nc")
tmin = xr.open_dataset("data/interim/tmin_up.nc")

# Merge them
ds = xr.merge([rainfall, tmax, tmin])

# Engineer features
engineer = FeatureEngineer()
ds = engineer.build(ds)

# Save
ds.to_netcdf("data/processed/climate_features.nc")

print("Saved data/processed/climate_features.nc")