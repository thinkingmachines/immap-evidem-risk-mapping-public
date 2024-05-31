# Import standard library modules
import os
from pathlib import Path

# The ROOT_DIR should represent the absolute path of the project root folder
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR_RAW = DATA_DIR / "raw"
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "src/config"
SQL_DIR = ROOT_DIR / "sql"
PROJ_CRS = "EPSG:4326"
LOCAL_CRS = "EPSG:3857"

# GCS ID and BQ Dataset IDs
GCP_PROJ_ID = "immap-evidem"
GCS_LANDSLIDES = "gs://immap-landslide-data"
GCS_VECTORS = "gs://immap-vectors"
GCS_RASTERS = "gs://immap-evidem-rasters"
GCS_ALIGNED = "gs://immap-aligned-data"

# SETS UP DATA DIRECTORY
# if exists, it will not overwrite
main_directories = ["admin_bounds", "aligned", "vectors", "rasters", "models", "output"]

aligned_subdirs = ["parquets", "csv"]
models_subdirs = ["training_data", "pkl", "rollout_data"]
output_subdirs = ["component_1", "component_2"]

for main_dirs in main_directories:
    path = DATA_DIR / main_dirs

    os.makedirs(path, exist_ok=True)

    if main_dirs == "aligned":
        for subdir in aligned_subdirs:
            sub_path = path / subdir
            sub_path.mkdir(exist_ok=True)

    if main_dirs == "models":
        for subdir in models_subdirs:
            sub_path = path / subdir
            sub_path.mkdir(exist_ok=True)

    if main_dirs == "output":
        for subdir in output_subdirs:
            sub_path = path / subdir
            sub_path.mkdir(exist_ok=True)
