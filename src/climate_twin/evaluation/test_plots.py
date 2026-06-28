import numpy as np

from climate_twin.evaluation.plots import ClimatePlots

truth = np.random.rand(129,135)

prediction = truth + np.random.normal(
    0,
    0.05,
    truth.shape
)

ClimatePlots.prediction(
    truth,
    prediction
)

ClimatePlots.error(
    truth,
    prediction
)

ClimatePlots.histogram(
    truth,
    prediction
)