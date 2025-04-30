# --------------------------------------
from pathlib import Path

# --------------------------------------
from environs import Env

# --------------------------------------
from streetscapes.sources import SourceType
from streetscapes.sources.image.base import ImageSourceBase


class KartaViewSource(ImageSourceBase):
    """TODO: Add docstrings"""
    @staticmethod
    def get_source_type() -> SourceType:
        """
        Get the enum corresponding to this source.
        """
        return SourceType.KartaView

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
    ):
        """
        An interface for downloading and manipulating
        street view images from Kartaview.

        Args:
            env:
                An Env object containing loaded configuration options.

            root_dir:
                An optional custom root directory. Defaults to None.
        """

        super().__init__(
            env,
            root_dir=root_dir,
            url=f"https://api.openstreetcam.org/2.0/photo",
        )

    def get_image_url(
        self,
        image_id: int | str,
    ) -> str:
        """
        Retrieve the URL for an image with the given ID.

        Args:

            image_id:
                The image ID.

        Returns:
            The URL to download the image.
        """

        url = f"{self.url}/?id={image_id}"
        try:
            # Send the request
            response = self.session.get(url)

            # Parse the response
            data = response.json()["result"]["data"][0]
            image_url = data["fileurlProc"]
            return image_url

        except Exception as e:
            return
