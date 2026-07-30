import xarray as xr
ds = xr.open_dataset("data/processed/climate_up_compressed.nc")
print(ds)