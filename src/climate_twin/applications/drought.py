import numpy as np

def compute_drought(rainfall, temperature):

    score = rainfall - 0.5*temperature

    drought = np.zeros_like(score)

    drought[score < -20] = 1
    drought[score < -40] = 2
    drought[score < -60] = 3

    return drought