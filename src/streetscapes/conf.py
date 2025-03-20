# --------------------------------------
import sys

# --------------------------------------
from pathlib import Path

# --------------------------------------
from loguru import logger

# --------------------------------------
from decouple import AutoConfig

# Setup
# ==================================================
# The root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent
config = AutoConfig(ROOT_DIR)

#: A local directory where data and output files are stored
LOCAL_DIR = ROOT_DIR / "local"

# Location of the Streetscapes data
# ==================================================
DATA_DIR = (
    Path(config("STREETSCAPES_DATA_DIR", LOCAL_DIR / "data"))
    .expanduser()
    .resolve()
    .absolute()
)

# Directory containing CSV, Parquet and image files.
# ==================================================
# Defaults to "DATA_DIR"
CSV_DIR = (
    Path(config("STREETSCAPES_CSV_DIR", DATA_DIR)).expanduser().resolve().absolute()
)

# Parquet files (the default mirrors the structure
# of the Global Streetscapes Huggingface repository).
PARQUET_DIR = (
    Path(config("STREETSCAPES_PARQUET_DIR", CSV_DIR / "parquet"))
    .expanduser()
    .resolve()
    .absolute()
)

# Images downloaded from Mapillary and KartaView
IMAGE_DIR = (
    Path(config("STREETSCAPES_IMAGE_DIR", LOCAL_DIR / "images"))
    .expanduser()
    .resolve()
    .absolute()
)


# Mapillary configuration
# ==================================================
MAPILLARY_TOKEN = config("MAPILLARY_TOKEN", None)

# Logger configuration
# ==================================================
# Enable colour tags in messages.
logger = logger.opt(colors=True)

#: Configurable log level.
LOG_LEVEL = config("STREETSCAPES_LOG_LEVEL", "INFO").upper()

#: Log format.
log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "<magenta>Streetscapes</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>",
            "level": LOG_LEVEL,
        }
    ]
}

logger.configure(**log_config)
