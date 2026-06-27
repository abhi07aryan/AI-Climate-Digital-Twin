from pathlib import Path
import xarray as xr


class DatasetWriter:

    def save(self, dataset: xr.Dataset, path: str):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dataset.to_netcdf(path)

        print(f"Dataset saved to {path}")