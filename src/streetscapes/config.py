import json
from pathlib import Path

CONFIG_FILE = Path.home() / ".config/streetscapes/config.json"
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "data_home": str(Path.home() / ".local/share/streetscapes"),
    "active_project": "streetscapes",
}


def initialize_config():
    """Create config file with defaults if it doesn’t exist."""
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2))


def load() -> dict:
    """Load config, initializing it if necessary."""
    initialize_config()
    cfg = json.loads(CONFIG_FILE.read_text())
    # Optional: fill in missing keys for forward compatibility
    for k, v in DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def get(key: str, default=None):
    return load().get(key, default)


def set(key: str, value):
    cfg = load()
    cfg[key] = value
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
