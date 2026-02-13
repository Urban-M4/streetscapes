import json
from omegaconf import OmegaConf
from platformdirs import user_config_path, user_cache_path

CONFIG_FILE = user_config_path("streetscapes", ensure_exists=True) / "config.json"


DEFAULTS = {
    "data_home": user_cache_path("streetscapes"),
    "active_project": "streetscapes",
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
    for k, v in DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def get(key: str, default=None):
    return load().get(key, default)


def set(key: str, value):
    cfg = load()
    cfg[key] = value
    OmegaConf.save(cfg, CONFIG_FILE)
