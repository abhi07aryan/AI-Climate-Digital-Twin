from climate_twin.data.rainfall_loader import RainfallLoader
from climate_twin.data.temperature_loader import TemperatureLoader
from climate_twin.data.shape_loader import ShapeLoader
from climate_twin.data.masking import SpatialMasker
from climate_twin.data.dataset_writer import DatasetWriter

from climate_twin.preprocessing.align import ClimateAligner
from climate_twin.preprocessing.merge import ClimateMerger
from climate_twin.preprocessing.features import FeatureEngineer


def main():

    print("Loading rainfall...")
    rainfall = RainfallLoader().load_all_years(
        "data/raw/imd/rainfall"
    )

    print("Loading Tmax...")
    tmax = TemperatureLoader("tmax").load_all_years(
        "data/raw/imd/temperature/max"
    )

    print("Loading Tmin...")
    tmin = TemperatureLoader("tmin").load_all_years(
        "data/raw/imd/temperature/min"
    )

    print("Aligning datasets...")
    aligner = ClimateAligner()

    tmax = aligner.align_to_reference(
        tmax,
        rainfall
    )

    tmin = aligner.align_to_reference(
        tmin,
        rainfall
    )

    print("Loading Uttar Pradesh boundary...")
    boundary = ShapeLoader().load(
        "data/raw/shapefiles/uttar_pradesh.shp"
    )

    print("Applying spatial mask...")
    masker = SpatialMasker()

    rainfall = masker.clip(rainfall, boundary)
    tmax = masker.clip(tmax, boundary)
    tmin = masker.clip(tmin, boundary)

    print("Merging datasets...")
    merger = ClimateMerger()

    climate = merger.merge([
        rainfall,
        tmax,
        tmin
    ])

    print("Engineering features...")
    engineer = FeatureEngineer()

    climate = engineer.build(climate)

    print("Saving processed dataset...")
    DatasetWriter().save(
        climate,
        "data/processed/climate_up.nc"
    )

    print("\nDataset successfully created!")
    print(climate)


if __name__ == "__main__":
    main()