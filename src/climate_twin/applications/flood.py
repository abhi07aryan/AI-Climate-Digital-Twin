import numpy as np

def compute_flood_risk(rainfall):

    risk = np.zeros_like(rainfall)

    risk[rainfall > 50] = 1
    risk[rainfall > 100] = 2
    risk[rainfall > 150] = 3

    return risk