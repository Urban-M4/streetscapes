# --------------------------------------
from decouple import AutoConfig

# --------------------------------------
from pathlib import Path

# Setup
# ==================================================
ROOT_DIR = Path(__file__).parent.parent
LOCAL_DIR = ROOT_DIR / "local"
config = AutoConfig(ROOT_DIR)

# Location of the Streetscapes data
# ==================================================
DATA_DIR = Path(
    config("STREETSCAPES_DATA_PATH", LOCAL_DIR / "streetscapes-data")
)
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Output directory
# ==================================================
OUTPUT_DIR = Path(config("STREETSCAPES_OUTPUT_DIR", LOCAL_DIR / "output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Mapillary configuration
# ==================================================
MAPILLARY_CLIENT_ID = config("MAPILLARY_CLIENT_ID")
MAPILLARY_TOKEN = config("MAPILLARY_TOKEN")
