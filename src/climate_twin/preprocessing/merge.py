import xarray as xr


class ClimateMerger:

    def merge(self, datasets):

        merged = xr.merge(datasets)

        return merged