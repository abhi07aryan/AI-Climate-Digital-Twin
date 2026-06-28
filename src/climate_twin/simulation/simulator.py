import torch

from climate_twin.forecasting.recursive_forecast import RecursiveForecaster
from climate_twin.simulation.scenario import ClimateScenario


class ClimateSimulator:
    """
    Runs climate simulations under different scenarios.
    """

    def __init__(self, model, device):

        self.model = model
        self.device = device

    def baseline(
        self,
        sequence,
        days=30
    ):
        """
        Forecast without changing any climate variables.
        """

        forecaster = RecursiveForecaster(
            self.model,
            self.device
        )

        forecast = forecaster.forecast(
            sequence,
            days
        )

        return forecast

    def warmer_world(
        self,
        sequence,
        delta_temp=2,
        days=30
    ):
        """
        Increase Tmax and Tmin by delta_temp.
        """

        scenario = ClimateScenario(sequence)

        modified = (
            scenario
            .increase_temperature(delta_temp)
            .get_sequence()
        )

        forecaster = RecursiveForecaster(
            self.model,
            self.device
        )

        return forecaster.forecast(
            modified,
            days
        )

    def wetter_world(
        self,
        sequence,
        rainfall_factor=1.5,
        days=30
    ):
        """
        Increase rainfall.
        """

        scenario = ClimateScenario(sequence)

        modified = (
            scenario
            .multiply_rainfall(rainfall_factor)
            .get_sequence()
        )

        forecaster = RecursiveForecaster(
            self.model,
            self.device
        )

        return forecaster.forecast(
            modified,
            days
        )

    def custom(
        self,
        sequence,
        modifications,
        days=30
    ):
        """
        Generic scenario.

        modifications example:

        {
            "temperature": +2,
            "rainfall_factor": 1.3
        }
        """

        scenario = ClimateScenario(sequence)

        if "temperature" in modifications:

            scenario.increase_temperature(
                modifications["temperature"]
            )

        if "rainfall_factor" in modifications:

            scenario.multiply_rainfall(
                modifications["rainfall_factor"]
            )

        forecaster = RecursiveForecaster(
            self.model,
            self.device
        )

        return forecaster.forecast(
            scenario.get_sequence(),
            days
        )