from pathlib import Path

import geopandas as gpd
import xarray as xr
up_boundary = gpd.read_file(
    "data/raw/shapefiles/UP_Boundary/UP_Boundary.shp"
)

up_boundary = up_boundary.to_crs("EPSG:4326")

DATASET = Path("data/processed/climate_up_compressed.nc")
ds =  xr.open_dataset(DATASET)
ds = ds.rio.write_crs("EPSG:4326")

ds = ds.rio.set_spatial_dims(
    x_dim="lon",
    y_dim="lat"
)

clipped = ds.rio.clip(
    up_boundary.geometry,
    up_boundary.crs,
    drop=True
)

clipped.to_netcdf(
    "data/processed/climate_up_compressed.nc"
)

print(clipped)