import xarray as xr
ds = xr.open_dataset("data/processed/climate_up_compressed.pnc")
print(ds)