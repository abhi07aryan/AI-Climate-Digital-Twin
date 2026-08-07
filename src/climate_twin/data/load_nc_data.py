from pathlib import Path

import xarray as xr

from climate_twin.data.shape_loader import ShapeLoader
from climate_twin.data.masking import SpatialMasker
from climate_twin.data.dataset_writer import DatasetWriter


# ----------------------------------------------------
# Paths
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAINFALL_DIR = DATA_DIR / "Rainfall"
TMAX_DIR = DATA_DIR / "max_temp" / "nc_files"
TMIN_DIR = DATA_DIR / "min_temp" / "nc_files"

BOUNDARY_FILE = (
    DATA_DIR
    / "shapefiles"
    / "UP_Boundary"
    / "UP_Boundary.shp"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"


# ----------------------------------------------------
# Standardize dataset
# ----------------------------------------------------

def standardize_dataset(ds, variable):

    # Rename coordinates if required
    coord_map = {}

    if "TIME" in ds.coords:
        coord_map["TIME"] = "time"

    if "LATITUDE" in ds.coords:
        coord_map["LATITUDE"] = "lat"

    if "LONGITUDE" in ds.coords:
        coord_map["LONGITUDE"] = "lon"

    if coord_map:
        ds = ds.rename(coord_map)

    # Rename the only data variable
    data_var = list(ds.data_vars)[0]
    ds = ds.rename({data_var: variable})

    return ds


# ----------------------------------------------------
# Load one climate variable
# ----------------------------------------------------

def load_variable(folder, variable, masker, boundary):

    files = sorted(folder.glob("*.nc"))

    if not files:
        raise FileNotFoundError(f"No .nc files found in {folder}")

    print(f"\nLoading {len(files)} {variable} files")

    datasets = []

    for file in files:

        print(f"Loading {file.name}")

        ds = xr.open_dataset(file)

        ds = standardize_dataset(ds, variable)

        ds = masker.clip(ds, boundary)

        datasets.append(ds)

    return xr.concat(
        datasets,
        dim="time",
        combine_attrs="override"
    )


# ----------------------------------------------------
# Main loader
# ----------------------------------------------------

def load_imd_data():

    boundary = ShapeLoader().load(BOUNDARY_FILE)

    masker = SpatialMasker()

    rainfall = load_variable(
        RAINFALL_DIR,
        "rainfall",
        masker,
        boundary,
    )

    tmax = load_variable(
        TMAX_DIR,
        "tmax",
        masker,
        boundary,
    )

    tmin = load_variable(
        TMIN_DIR,
        "tmin",
        masker,
        boundary,
    )

    return rainfall, tmax, tmin


# ----------------------------------------------------
# Script
# ----------------------------------------------------

if __name__ == "__main__":

    rainfall, tmax, tmin = load_imd_data()

    writer = DatasetWriter()

    writer.save(
        rainfall,
        OUTPUT_DIR / "rainfall_up.nc"
    )

    writer.save(
        tmax,
        OUTPUT_DIR / "tmax_up.nc"
    )

    writer.save(
        tmin,
        OUTPUT_DIR / "tmin_up.nc"
    )

    print("\nPreprocessing completed successfully.")