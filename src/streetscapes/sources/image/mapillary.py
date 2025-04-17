# --------------------------------------
from pathlib import Path

# --------------------------------------
import requests

# --------------------------------------
from environs import Env

# --------------------------------------
import json

# --------------------------------------
from streetscapes.sources import SourceType
from streetscapes.sources.image.base import ImageSourceBase


class MapillarySource(ImageSourceBase):

    @staticmethod
    def get_source_type() -> SourceType:
        """
        Get the enum corresponding to this source.
        """
        return SourceType.Mapillary

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
    ):
        """
        An interface for downloading and manipulating
        street view images from Mapillary.

        Args:

            env:
                An Env object containing loaded configuration options.

            root_dir:
                An optional custom root directory. Defaults to None.
        """

        super().__init__(
            env,
            root_dir=root_dir,
            url=f"https://graph.mapillary.com",
        )

    def get_image_url(
        self,
        image_id: int | str,
    ) -> str | None:
        """
        Retrieve the URL for an image with the given ID.

        Args:
            image_id:
                The image ID.

        Returns:
            str:
                The URL to query.
        """

        url = f"{self.url}/{image_id}?fields=thumb_original_url"

        rq = requests.Request("GET", url, params={"access_token": self.token})
        res = self.session.send(rq.prepare())
        if res.status_code == 200:
            return json.loads(res.content.decode("utf-8"))[
                f"thumb_original_url"
            ]

    def create_session(self) -> requests.Session:
        """
        Create an (authenticated) session for the supplied source.

        Returns:
            A `requests` session.
        """

        session = requests.Session()
        session.headers.update({"Authorization": f"OAuth {self.token}"})
        return session
