from climate_twin.data.imd_base import IMDBaseLoader

import numpy as np


class RainfallLoader(IMDBaseLoader):
    """
    Loader for IMD Daily Rainfall
    (0.25° × 0.25° gridded dataset).
    """

    VARIABLE_NAME = "rainfall"

    N_LAT = 129
    N_LON = 135

    LAT_START = 6.5
    LON_START = 66.5

    RESOLUTION = 0.25

    DTYPE = np.float32

    def handle_missing(self, data):
        """
        Replace IMD rainfall missing values with NaN.
        """

        data = data.astype(np.float32)

        # IMD rainfall missing value
        data[data < -900] = np.nan

        return data