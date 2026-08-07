from climate_twin.data.imd_base import IMDBaseLoader

import numpy as np

class TemperatureLoader(IMDBaseLoader):

    def __init__(self, variable):
        self.VARIABLE_NAME = variable
        super().__init__()

    def handle_missing(self, data):

        data = data.where(data != 99.9)

        return data.interpolate_na(
            dim="time",
            method="linear"
        )