import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        # Ignore NaN pixels
        mask = ~torch.isnan(target)

        prediction = prediction[mask]
        target = target[mask]

        return torch.mean((prediction - target) ** 2)