import numpy as np


class ClimateScenario:
    """
    Apply climate change scenarios to an input sequence.

    Sequence shape:
        (window, channels, height, width)
    """

    def __init__(self, sequence):

        self.sequence = sequence.copy()

    def increase_temperature(self, delta):

        # Tmax
        self.sequence[:, 1] += delta

        # Tmin
        self.sequence[:, 2] += delta

        # Mean temperature
        self.sequence[:, 3] += delta

        return self

    def decrease_temperature(self, delta):

        return self.increase_temperature(-delta)

    def multiply_rainfall(self, factor):

        # Rainfall
        self.sequence[:, 0] *= factor

        # Rolling rainfall
        self.sequence[:, 5] *= factor
        self.sequence[:, 6] *= factor

        # Rain anomaly
        self.sequence[:, 9] *= factor

        return self

    def add_rainfall(self, amount):

        self.sequence[:, 0] += amount

        return self

    def scale_feature(self, channel, factor):

        self.sequence[:, channel] *= factor

        return self

    def get_sequence(self):

        return self.sequence