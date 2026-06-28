import numpy as np


class ClimateScenario:
    """
    Modify climate input variables before forecasting.
    """

    def __init__(self, sequence):

        # shape:
        # (window, channels, H, W)

        self.sequence = sequence.copy()

    def increase_temperature(self, delta):

        # channel 1 = Tmax
        # channel 2 = Tmin

        self.sequence[:, 1] += delta
        self.sequence[:, 2] += delta

        return self

    def decrease_temperature(self, delta):

        self.sequence[:, 1] -= delta
        self.sequence[:, 2] -= delta

        return self

    def multiply_rainfall(self, factor):

        # channel 0 = rainfall

        self.sequence[:, 0] *= factor

        return self

    def add_rainfall(self, amount):

        self.sequence[:, 0] += amount

        return self

    def scale_feature(self, channel, factor):

        self.sequence[:, channel] *= factor

        return self

    def get_sequence(self):

        return self.sequence