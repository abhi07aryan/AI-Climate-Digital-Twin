import numpy as np

def compute_flood_risk(rainfall):
    """
    Flood Risk Classification

    0 = No Risk
    1 = Low Risk
    2 = Moderate Risk
    3 = High Risk
    """

    rainfall = np.asarray(rainfall)

    risk = np.zeros(rainfall.shape, dtype=np.uint8)

    risk[(rainfall > 0.2) & (rainfall <= 0.5)] = 1
    risk[(rainfall > 0.5) & (rainfall <= 1.0)] = 2
    risk[rainfall > 1.0] = 3

    return risk