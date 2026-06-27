from pathlib import Path
from pydantic import BaseModel
import yaml


class Project(BaseModel):
    name: str
    version: str


class Region(BaseModel):
    country: str
    pilot_state: str


class Paths(BaseModel):
    rainfall: str
    tmax: str
    tmin: str

    lst: str
    sst: str
    insat_rainfall: str

    india_boundary: str
    up_boundary: str

    interim: str
    processed: str


class Processing(BaseModel):
    target_resolution: float
    interpolation: str


class Forecast(BaseModel):
    sequence_length: int
    horizon: int


class Settings(BaseModel):
    project: Project
    region: Region
    paths: Paths
    processing: Processing
    forecast: Forecast


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

settings = Settings(**cfg)