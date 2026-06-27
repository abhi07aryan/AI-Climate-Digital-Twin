from pathlib import Path
import geopandas as gpd

from climate_twin.logger import logger


class ShapeLoader:

    def load(self, filepath):

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(filepath)

        logger.info(f"Loading {filepath.name}")

        gdf = gpd.read_file(filepath)

        logger.info(f"{len(gdf)} feature(s) loaded.")

        return gdf