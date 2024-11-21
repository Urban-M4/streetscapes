# --------------------------------------
import sys

# --------------------------------------
import logging

# --------------------------------------
from pathlib import Path

# --------------------------------------
from loguru import logger

# --------------------------------------
from decouple import AutoConfig

logger = logging.getLogger("mapillary.utils.client")
logger.setLevel(logging.ERROR)

# Setup
# ==================================================
# The root directory of the project
ROOT_DIR = Path(__file__).parent.parent
config = AutoConfig(ROOT_DIR)

# A local directory where data and output files
LOCAL_DIR = ROOT_DIR / "local"

# Location of the Streetscapes data
# ==================================================
DATA_DIR = (
    Path(config("STREETSCAPES_DATA_DIR", LOCAL_DIR / "streetscapes-data"))
    .expanduser()
    .resolve()
    .absolute()
)

# Output directory
# ==================================================
OUTPUT_DIR = (
    Path(config("STREETSCAPES_OUTPUT_DIR", LOCAL_DIR / "output"))
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
