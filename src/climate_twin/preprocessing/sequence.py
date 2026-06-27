import numpy as np


class SequenceBuilder:

    def __init__(self, window=30):

        self.window = window

    def create(self, ds, target="rainfall"):

        X = []

        y = []

        values = ds.to_array().values

        values = np.moveaxis(values, 0, -1)

        for i in range(

            len(ds.time) - self.window

        ):

            X.append(

                values[
                    i:i+self.window
                ]

            )

            y.append(

                ds[target]
                .isel(time=i+self.window)
                .values

            )

        return np.array(X), np.array(y)