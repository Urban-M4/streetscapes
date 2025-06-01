# --------------------------------------
from pathlib import Path

# --------------------------------------
from environs import Env

# --------------------------------------
from streetscapes.sources.image.base import ImageSourceBase


class Google(ImageSourceBase):
    """TODO: Add docstrings"""

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
    ):
        """
        An interface for downloading and manipulating
        street view images from Google StreetView.

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
