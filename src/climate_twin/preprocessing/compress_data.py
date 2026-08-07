import xarray as xr

# Open your existing dataset
ds = xr.open_dataset("data/processed/climate_features.nc")
# (Optional) Convert to float32 to reduce size
for var in ds.data_vars:
    if ds[var].dtype == "float64":
        ds[var] = ds[var].astype("float32")

# Compression settings
encoding = {
    var: {
        "zlib": True,
        "complevel": 5,
    }
    for var in ds.data_vars
}

# Save compressed dataset
ds.to_netcdf(
    "data/processed/climate_up_compressed.nc",
    encoding=encoding,
)

print("Compression complete!")