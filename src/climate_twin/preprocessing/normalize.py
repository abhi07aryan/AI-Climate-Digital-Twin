from sklearn.preprocessing import StandardScaler
import numpy as np


class ClimateNormalizer:

    def __init__(self):
        self.scalers = {}

    def fit_transform(self, ds, variables):

        ds = ds.copy()

        for variable in variables:

            data = ds[variable].values

            mask = np.isnan(data)

            valid = data[~mask].reshape(-1, 1)

            scaler = StandardScaler()

            valid_scaled = scaler.fit_transform(valid)

            data_scaled = data.copy()
            data_scaled[~mask] = valid_scaled.ravel()

            ds[variable].values = data_scaled

            self.scalers[variable] = scaler

        return ds