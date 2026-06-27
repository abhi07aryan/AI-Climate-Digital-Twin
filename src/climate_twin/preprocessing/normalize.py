import xarray as xr


class ClimateNormalizer:

    def __init__(self):
        self.stats = {}

    def fit(self, ds: xr.Dataset, variables):

        for var in variables:

            self.stats[var] = {
                "mean": ds[var].mean(skipna=True),
                "std": ds[var].std(skipna=True)
            }

    def transform(self, ds: xr.Dataset):

        ds = ds.copy()

        for var, stat in self.stats.items():

            ds[var] = (
                ds[var] - stat["mean"]
            ) / stat["std"]

        return ds

    def fit_transform(self, ds: xr.Dataset, variables):

        self.fit(ds, variables)

        return self.transform(ds)