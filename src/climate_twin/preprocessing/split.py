import xarray as xr


class TimeSeriesSplit:
    """
    Split climate dataset chronologically.
    """

    def __init__(
        self,
        train_end="2020-12-31",
        valid_end="2022-12-31",
    ):

        self.train_end = train_end
        self.valid_end = valid_end

    def split(self, ds: xr.Dataset):

        train = ds.sel(
            time=slice(None, self.train_end)
        )

        valid = ds.sel(
            time=slice(
                "2021-01-01",
                self.valid_end
            )
        )

        test = ds.sel(
            time=slice(
                "2023-01-01",
                None
            )
        )

        return train, valid, test