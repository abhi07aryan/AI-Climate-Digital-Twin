import xarray as xr


class TimeSeriesSplit:
    """
    Chronological train/validation/test split.
    """

    def __init__(
        self,
        train_end="2020-12-31",
        valid_start="2021-01-01",
        valid_end="2022-12-31",
        test_start="2023-01-01",
    ):

        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start

    def split(self, ds: xr.Dataset):

        train = ds.sel(
            time=slice(None, self.train_end)
        )

        valid = ds.sel(
            time=slice(
                self.valid_start,
                self.valid_end
            )
        )

        test = ds.sel(
            time=slice(
                self.test_start,
                None
            )
        )

        print(f"Train : {train.time.size} days")
        print(f"Valid : {valid.time.size} days")
        print(f"Test  : {test.time.size} days")

        return train, valid, test