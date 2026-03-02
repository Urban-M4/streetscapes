import os
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
from streetscapes.utils import ensure_dir
from platformdirs import user_config_path, user_cache_path, user_data_path


class Configuration(BaseSettings):
    config_file: Path = user_config_path("streetscapes") / "config.json"
    project_dir: Path = user_data_path("streetscapes")
    image_dir: Path = user_cache_path("streetscapes")
    active_project: str = "streetscapes"
    mapillary_token: str = os.getenv("MAPILLARY_TOKEN", "")

    @field_validator("project_dir", mode="before")
    @classmethod
    def _ensure_project_dir(cls, value: str | Path) -> Path:
        return ensure_dir(value)

    @field_validator("image_dir", mode="before")
    @classmethod
    def _ensure_image_dir(cls, value: str | Path) -> Path:
        return ensure_dir(value)

    @field_validator("config_file", mode="before")
    @classmethod
    def _ensure_config_file(cls, value: str | Path) -> Path:
        value = Path(value)
        Path.touch(value)
        return value

    def save(self):
        self.config_file.write_text(self.model_dump_json(indent=4))


conf = Configuration()
