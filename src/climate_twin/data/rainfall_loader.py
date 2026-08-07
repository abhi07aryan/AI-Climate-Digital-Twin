from climate_twin.data.imd_base import IMDBaseLoader

import numpy as np

class RainfallLoader(IMDBaseLoader):

    VARIABLE_NAME = "rainfall"

    def handle_missing(self, data):

        data = data.where(data > -900)

        return data.interpolate_na(
            dim="time",
            method="linear"
        )