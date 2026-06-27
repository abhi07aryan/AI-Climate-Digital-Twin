import regionmask # type: ignore
import xarray as xr


class SpatialMasker:
    """
    Spatially mask a climate dataset using a state boundary.
    """

    def clip(self, dataset: xr.Dataset, boundary):

        boundary = boundary.to_crs("EPSG:4326")

        region = regionmask.from_geopandas(boundary)

        mask = region.mask(dataset.lon, dataset.lat)

        clipped = dataset.where(mask == 0)

        return clipped