import numpy as np

def compute_drought(rainfall, tmax, tmin, normalizer):
    """
    Drought Classification

    0 = No drought
    1 = Mild
    2 = Moderate
    3 = Severe
    """

    # Convert back to physical units
    rainfall = normalizer.inverse_transform_array(
        rainfall,
        "rainfall"
    )

    tmax = normalizer.inverse_transform_array(
        tmax,
        "tmax"
    )

    tmin = normalizer.inverse_transform_array(
        tmin,
        "tmin"
    )

    # Mean temperature
    tmean = (tmax + tmin) / 2

    # Simple drought score
    score = tmean - rainfall

    # Dynamic thresholds
    q50 = np.percentile(score, 50)
    q75 = np.percentile(score, 75)
    q90 = np.percentile(score, 90)

    drought = np.zeros_like(score, dtype=np.uint8)

    drought[score >= q50] = 1
    drought[score >= q75] = 2
    drought[score >= q90] = 3

    return drought