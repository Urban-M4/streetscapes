import os
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
from streetscapes.utils import ensure_dir
from platformdirs import user_config_path, user_cache_path, user_data_path

CONFIG_FILE = user_config_path("streetscapes", ensure_exists=True) / "config.json"


class Configuration(BaseSettings):
    project_dir: Path = user_data_path("streetscapes")
    image_dir: Path = user_cache_path("streetscapes")
    active_project: str = "streetscapes"
    mapillary_token: str = os.getenv("MAPILLARY_TOKEN", "")

    @field_validator("project_dir", "image_dir", mode="before")
    @classmethod
    def _ensure_dir(cls, value: str | Path) -> Path:
        return ensure_dir(value)


conf = Configuration(CONFIG_FILE)
