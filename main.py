import xarray as xr
from pathlib import Path
import gdown

ds = Path("data/processed/climate_up_clip.nc")
def download_dataset():
    ds.parent.mkdir(parents=True, exist_ok=True)

    file_id = "1sZeQ45vGjq-7xx1RfLeoBDg_ZQqHmSWT"

    gdown.download(
        id=file_id,
        output=str(ds),
        quiet=False
    )

if not ds.exists():
    download_dataset()

print(ds)