import os
from omegaconf import OmegaConf
from platformdirs import user_config_path, user_cache_path, user_data_path

CONFIG_FILE = user_config_path("streetscapes", ensure_exists=True) / "config.json"


DEFAULTS = {
    "project_dir": user_data_path("streetscapes"),
    "image_dir": user_cache_path("streetscapes"),
    "active_project": "streetscapes",
    "mapillary_token": os.getenv("MAPILLARY_TOKEN", ""),
}


def initialize_config():
    """Create config file with defaults if it doesn’t exist."""
    if not CONFIG_FILE.exists():
        cfg = OmegaConf.create(DEFAULTS)
        OmegaConf.save(cfg, CONFIG_FILE)


def load() -> dict:
    """Load config, initializing it if necessary."""
    initialize_config()
    cfg = OmegaConf.load(CONFIG_FILE)
    # Optional: fill in missing keys for forward compatibility
    new_keys = set(DEFAULTS.keys()).difference(set(cfg.keys()))
    for k in new_keys:
        cfg.setdefault(k, DEFAULTS[k])

    # Optional: remove deprecated keys that are not in DEFAULTS
    deprecated_keys = set(cfg.keys()).difference(set(DEFAULTS.keys()))
    for k in deprecated_keys:
        cfg.pop(k)

    return cfg


def getopt(key: str, default=None):
    return load().get(key, default)


def setopt(key: str, value):
    cfg = load()
    if key in DEFAULTS:
        cfg[key] = value
        OmegaConf.save(cfg, CONFIG_FILE)
