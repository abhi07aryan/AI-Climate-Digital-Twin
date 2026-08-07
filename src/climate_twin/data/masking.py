import numpy as np
import regionmask
import xarray as xr

class SpatialMasker:

    def clip(self, dataset: xr.Dataset, boundary):

        boundary = boundary.to_crs("EPSG:4326")

        var = list(dataset.data_vars)[0]

        # Crop
        minx, miny, maxx, maxy = boundary.total_bounds

        dataset = dataset.sel(
            lon=slice(minx, maxx),
            lat=slice(miny, maxy)
        )
        # Region mask
        region = regionmask.from_geopandas(boundary)

        mask = region.mask(dataset.lon, dataset.lat)

        print("Mask valid:", np.isfinite(mask.values).sum())

        dataset = dataset.where(mask.notnull())

        return dataset