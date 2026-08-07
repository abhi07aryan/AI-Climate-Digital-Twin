import regionmask
import xarray as xr

class SpatialMasker:

    def clip(self, dataset: xr.Dataset, boundary):

        boundary = boundary.to_crs("EPSG:4326")

        # Crop to bounding box
        minx, miny, maxx, maxy = boundary.total_bounds

        dataset = dataset.sel(
            lon=slice(minx, maxx),
            lat=slice(miny, maxy)
        )

        # Mask outside the state
        region = regionmask.from_geopandas(boundary)

        mask = region.mask(dataset.lon, dataset.lat)

        return dataset.where(mask.notnull())