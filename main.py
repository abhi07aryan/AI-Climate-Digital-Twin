from climate_twin.data.rainfall_loader import RainfallLoader
from climate_twin.data.temperature_loader import TemperatureLoader
from climate_twin.data.shape_loader import ShapeLoader
from climate_twin.data.masking import SpatialMasker

from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.preprocessing.align import ClimateAligner
from climate_twin.preprocessing.merge import ClimateMerger
from climate_twin.preprocessing.features import FeatureEngineer
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.sequence import SequenceBuilder


def main():

    # ----------------------------------------------------
    # Load datasets
    # ----------------------------------------------------

    rainfall = RainfallLoader().load_all_years(
        "data/raw/imd/rainfall"
    )

    tmax = TemperatureLoader("tmax").load_all_years(
        "data/raw/imd/temperature/max"
    )

    tmin = TemperatureLoader("tmin").load_all_years(
        "data/raw/imd/temperature/min"
    )

    # ----------------------------------------------------
    # Align temperature to rainfall grid
    # ----------------------------------------------------

    aligner = ClimateAligner()

    tmax = aligner.align_to_reference(
        tmax,
        rainfall
    )

    tmin = aligner.align_to_reference(
        tmin,
        rainfall
    )

    # ----------------------------------------------------
    # Load Uttar Pradesh boundary
    # ----------------------------------------------------

    boundary = ShapeLoader().load(
        "data/raw/shapefiles/uttar_pradesh.shp"
    )

    # ----------------------------------------------------
    # Mask datasets
    # ----------------------------------------------------

    masker = SpatialMasker()

    rainfall = masker.clip(rainfall, boundary)

    tmax = masker.clip(tmax, boundary)

    tmin = masker.clip(tmin, boundary)

    # ----------------------------------------------------
    # Merge
    # ----------------------------------------------------

    merger = ClimateMerger()

    climate = merger.merge([
        rainfall,
        tmax,
        tmin
    ])

    # ----------------------------------------------------
    # Feature Engineering
    # ----------------------------------------------------

    engineer = FeatureEngineer()

    climate = engineer.build(climate)

    # ----------------------------------------------------
    # Normalize
    # ----------------------------------------------------

    normalizer = ClimateNormalizer()

    variables = [

        "rainfall",

        "tmax",

        "tmin",

        "temp_mean",

        "temp_range",

        "rain_7day",

        "rain_30day"

    ]

    climate = normalizer.fit_transform(
        climate,
        variables
    )

    # ----------------------------------------------------
    # Verification
    # ----------------------------------------------------

    splitter = TimeSeriesSplit()
    train, valid, test = splitter.split(climate)
    print()
    builder = SequenceBuilder(
    window_size=30,
    target="rainfall"
    )

    X_train, y_train = builder.build(train)

    X_valid, y_valid = builder.build(valid)

    X_test, y_test = builder.build(test)
    print(X_train.shape)
    print(y_train.shape)

if __name__ == "__main__":
    main()