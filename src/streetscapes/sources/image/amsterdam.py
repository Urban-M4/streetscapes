# --------------------------------------
from pathlib import Path

# --------------------------------------
from environs import Env

# --------------------------------------
from streetscapes.sources import SourceType
from streetscapes.sources.image.base import ImageSourceBase


class AmsterdamSource(ImageSourceBase):
    """TODO: Add docstrings"""
    @staticmethod
    def get_source_type() -> SourceType:
        """
        Get the enum corresponding to this source.
        """
        return SourceType.Amsterdam

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
    ):
        """
        An interface for downloading and manipulating
        street view images from the Amsterdam repository.

        Args:
            env:
                An Env object containing loaded configuration options.

            root_dir:
                An optional custom root directory. Defaults to None.
        """

        super().__init__(
            env,
            root_dir=root_dir,
            url=None,
        )
