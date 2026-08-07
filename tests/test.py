from pathlib import Path

import xarray as xr

DATASET = Path("data/processed/climate_up_clip.nc")
ds =  xr.open_dataset(DATASET)

print(ds["rainfall"].max().item())