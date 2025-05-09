# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from environs import Env

# --------------------------------------
from streetscapes.sources.image.base import ImageSourceBase

class KartaView(ImageSourceBase):
    """TODO: Add docstrings"""

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

    def fetch_image_ids(self, lat, lon, radius):
        """
        Fetch Kartaview image ids within radius of a given point.

        Uses old openstreecam API (https://api.openstreetcam.org/api/doc.html) as it seems not available on latest kartaview
        API (https://doc.kartaview.org/#section/API-Resources).

        Returns:
            pd.DataFrame
        """
        page = 1
        photos = []
        while True:
            resp = self.session.post(
                "https://api.openstreetcam.org/1.0/list/nearby-photos/",
                files={
                    "lat": (None, str(lat)),
                    "lng": (None, str(lon)),
                    "radius": (None, str(radius)),
                    "page": (None, str(page)),
                },
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("currentPageItems", [])
            if not items:
                break

            photos.extend(items)
            page += 1

        return ibis.memtable(photos)
