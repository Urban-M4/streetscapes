# --------------------------------------
import sys

# --------------------------------------
from loguru import logger

# Logger configuration
# ==================================================
# Enable colour tags in messages.
logger = logger.opt(colors=True)

#: Log format.
log_config = {
    "handlers": [
        {
            "sink": sys.stderr,
            "format": "<magenta>Streetscapes</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>",
            "level": "INFO",
            "enqueue": True,
            "diagnose": False,
        }
    ]
}

logger.configure(**log_config)
