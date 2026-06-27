import torch
from torch.utils.data import Dataset
import numpy as np


class ClimateTorchDataset(Dataset):
    """
    PyTorch Dataset for climate forecasting.
    """

    def __init__(
        self,
        ds,
        input_features,
        target="rainfall",
        window_size=30,
    ):

        self.window = window_size
        self.target = target

        self.features = input_features

        # Shape:
        # (time, feature, lat, lon)

        self.X = (
            ds[input_features]
            .to_array()
            .transpose(
                "time",
                "variable",
                "lat",
                "lon"
            )
            .values
        )

        self.y = ds[target].values

    def __len__(self):

        return len(self.y) - self.window

    def __getitem__(self, idx):

        x = np.nan_to_num(
            self.X[idx:idx+self.window],
            nan=0.0
        )

        y = np.nan_to_num(
            self.y[idx+self.window],
            nan=0.0
        )
        return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0))