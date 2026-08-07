import xarray as xr

ds = xr.open_dataset("data/raw/max_temp/nc_files/Maxtemp_MaxT_1951.nc")

print(ds)
print(ds.coords)