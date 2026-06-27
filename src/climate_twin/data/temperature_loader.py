from climate_twin.data.imd_base import IMDBaseLoader

import numpy as np


class TemperatureLoader(IMDBaseLoader):

    def __init__(self, variable="tmax"):

        self.VARIABLE_NAME = variable

        self.N_LAT = 31
        self.N_LON = 31

        self.LAT_START = 7.5
        self.LON_START = 67.5

        self.RESOLUTION = 1.0

        self.DTYPE = np.float32

    def handle_missing(self, data):

        data[np.isclose(data, 99.9)] = np.nan

        return data