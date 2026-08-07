from pathlib import Path
import xarray as xr
import numpy as np

DATASET = Path("data/processed/climate_up_compressed.nc")

ds = xr.open_dataset(DATASET)

image = ds["temp_range"].sel(time="1951-01-01")
import numpy as np

arr = ds["tmax"].sel(time="1951-01-01").values

print(arr.shape)
print(np.where(np.isfinite(arr)))
