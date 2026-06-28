import numpy as np

from climate_twin.simulation.compare import ClimateComparison

baseline = np.random.rand(
    5,
    129,
    135
)

scenario = baseline + np.random.normal(
    0.2,
    0.05,
    baseline.shape
)

compare = ClimateComparison()

compare.print_statistics(
    baseline,
    scenario
)

compare.plot_day(
    baseline,
    scenario,
    day=0
)