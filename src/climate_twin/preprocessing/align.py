import xarray as xr


class ClimateAligner:
    """
    Align multiple climate datasets to a common grid.
    """

    def align_to_reference(self, source, reference):

        aligned = source.interp(

            lat=reference.lat,
            lon=reference.lon,

            method="linear"

        )

        return aligned