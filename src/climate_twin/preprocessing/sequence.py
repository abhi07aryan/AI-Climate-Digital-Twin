import numpy as np
import xarray as xr


class SequenceBuilder:

    def __init__(self, window_size=30, target="rainfall"):

        self.window_size = window_size
        self.target = target

    def build(self, ds: xr.Dataset):

        features = list(ds.data_vars)

        X = ds[features].to_array()

        X = X.transpose(
            "time",
            "lat",
            "lon",
            "variable"
        )

        X = X.values

        y = ds[self.target].values

        X_seq = []
        y_seq = []

        for i in range(len(ds.time) - self.window_size):

            X_seq.append(
                X[i:i + self.window_size]
            )

            y_seq.append(
                y[i + self.window_size]
            )

        return (
            np.array(X_seq),
            np.array(y_seq)
        )