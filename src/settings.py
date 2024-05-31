# Import standard library modules
from pathlib import Path

# The ROOT_DIR should represent the absolute path of the project root folder
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "src/config"
PROJ_CRS = "EPSG:4326"

# GCS ID and BQ Dataset IDs
GCP_PROJ_ID = "tm-geospatial"
