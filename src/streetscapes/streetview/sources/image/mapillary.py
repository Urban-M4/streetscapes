# --------------------------------------
import requests

# --------------------------------------
from pathlib import Path

# --------------------------------------
import json

# --------------------------------------
from streetscapes import logger
from streetscapes.streetview.workspace import SVWorkspace
from streetscapes.streetview.sources.image.base import ImageSourceBase


class MapillarySource(ImageSourceBase):

    def __init__(
        self,
        workspace: SVWorkspace,
        resolution: int = 1024,
    ):
        super().__init__(
            workspace,
            url=f"https://graph.mapillary.com",
            name="MAPILLARY",
        )

        # Image resolution to request.
        self.resolution = resolution

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
            str:
                The URL to query.
        """

        url = f"{self.url}/{image_id}?fields=thumb_{self.resolution}_url"

        try:
            rq = requests.Request("GET", url, params={"access_token": self.token})
            res = self.session.send(rq.prepare())
            image_url = json.loads(res.content.decode("utf-8"))[
                f"thumb_{self.resolution}_url"
            ]

            return image_url

        except Exception as e:
            logger.info(f"Error: {e}")
            return


    def create_session(self) -> requests.Session:
        """
        Create an (authenticated) session for the supplied source.

        Returns:
            A `requests` session.
        """

        session = requests.Session()
        session.headers.update({"Authorization": f"OAuth {self.token}"})
        return session
