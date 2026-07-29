import xarray as xr
from pathlib import Path
import gdown

ds = Path("data/processed/climate_up.nc")
def download_dataset():
    ds.parent.mkdir(parents=True, exist_ok=True)

    file_id = "1Ld7oVZJ5XCFi6o8ZPZ0iM9vErvmQTmZu"

    gdown.download(
        id=file_id,
        output=str(ds),
        quiet=False
    )

if not ds.exists():
    download_dataset()

print(ds)