import numpy as np

def compute_drought(rainfall, tmax, tmin):
    """
    Continuous drought index

    0 = No drought
    1 = Mild
    2 = Moderate
    3 = Severe
    """

    tmean = (tmax + tmin) / 2

    rainfall = np.maximum(rainfall, 1)

    dryness = (tmean / rainfall)

    drought = np.zeros_like(dryness, dtype=np.uint8)

    drought[dryness > 1.0] = 1
    drought[dryness > 2.0] = 2
    drought[dryness > 3.0] = 3

    return drought