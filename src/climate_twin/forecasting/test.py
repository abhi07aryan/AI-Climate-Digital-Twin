import numpy as np

from climate_twin.forecasting.save_forecast import ForecastWriter

forecast = np.random.rand(
    30,
    129,
    135
)

lat = np.arange(7.5, 36.5)

lon = np.arange(67.5, 97.5)

writer = ForecastWriter()

writer.save(

    forecast,

    lat,

    lon,

    start_date="2025-01-01"

)