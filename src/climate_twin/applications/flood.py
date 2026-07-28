import numpy as np

def compute_flood_risk(rainfall, normalizer):

    rainfall = normalizer.inverse_transform_array(
        rainfall,
        "rainfall"
    )

    risk = np.zeros_like(rainfall, dtype=np.uint8)

    risk[(rainfall >= 20) & (rainfall < 50)] = 1
    risk[(rainfall >= 50) & (rainfall < 100)] = 2
    risk[rainfall >= 100] = 3

    return risk