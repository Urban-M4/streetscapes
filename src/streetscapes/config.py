"""Streetscapes config handler."""

import os
from pathlib import Path

import orjson as oj
from platformdirs import user_cache_path, user_config_path, user_data_path
from pydantic import field_validator
from pydantic_settings import BaseSettings

from streetscapes.utils import ensure_dir

CONFIG_FILE = user_config_path("streetscapes", ensure_exists=True) / "config.json"


class Configuration(BaseSettings):
    project_dir: Path = user_data_path("streetscapes")
    image_dir: Path = user_cache_path("streetscapes")
    active_project: str = "streetscapes"
    mapillary_token: str = os.getenv("MAPILLARY_TOKEN", "")
    local_cache_dir_name: str = "local"
    sam3_model_path: Path | None = None

    @field_validator("project_dir", mode="before")
    @classmethod
    def _ensure_project_dir(cls, value: str | Path) -> Path:
        return ensure_dir(value)

    @field_validator("image_dir", mode="before")
    @classmethod
    def _ensure_image_dir(cls, value: str | Path) -> Path:
        return ensure_dir(value)

    @field_validator("sam3_model_path", mode="after")
    @classmethod
    def _validate_sam3_model_path(cls, value: Path | None) -> Path | None:
        if value is not None:
            if not value.is_absolute():
                raise ValueError(f"SAM3 model path must be absolute: {value}")
            if not value.is_file():
                raise ValueError(f"SAM3 model path does not exist: {value}")
        return value

    def save(self):
        """Save configuration to file."""
        CONFIG_FILE.write_text(self.model_dump_json(indent=4))


if not CONFIG_FILE.exists():
    Configuration().save()

CFG = Configuration(**oj.loads(Path.read_text(CONFIG_FILE)))
