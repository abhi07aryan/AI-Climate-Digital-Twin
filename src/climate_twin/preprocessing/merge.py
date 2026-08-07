import xarray as xr

rainfall = xr.open_dataset("data/interim/rainfall_up.nc")
tmax = xr.open_dataset("data/interim/tmax_up.nc")
tmin = xr.open_dataset("data/interim/tmin_up.nc")

# Merge all variables
ds = xr.merge([rainfall, tmax, tmin], compat="override")

print(ds)

ds.to_netcdf("data/processed/climate_up.nc")