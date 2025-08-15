# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from streetscapes.sources.image.base import ImageSourceBase

class AmsterdamPanorama(ImageSourceBase):
    """TODO: Add docstrings"""

    def __init__(
        self,
        root_dir: str | Path | None = None,
    ):
        """
        An interface for downloading and manipulating
        street view images from the Amsterdam repository.

        Args:
            root_dir:
                An optional custom root directory. Defaults to
                DATA_HOME/sources/amsterdampanorama, where DATA_HOME is read from the
                environment variables. Defaults to None.
        """

        super().__init__(
            root_dir=root_dir,
            url="https://api.data.amsterdam.nl/panorama/panoramas/",
        )

    def get_image_url(self, image_id):
        raise NotImplementedError(
            "get_image_url not implemented for Amsterdam. Use URLs returned by fetch_image_ids directly."
        )

    def fetch_image_ids(self, lat, lon, radius):
        return self._fetch_image_ids(near=f"{lon},{lat}", radius=radius)

    def _fetch_image_ids(self, **params):
        """Fetch panorama IDs and metadata within radius of given point."""
        panoramas = []

        response = self.session.get(self.url, params=params)
        while True:
            response.raise_for_status()
            data = response.json()
            # Process results
            for pano in data["_embedded"]["panoramas"]:
                panoramas.append(
                    {
                        "pano_id": pano["pano_id"],
                        "timestamp": pano["timestamp"],
                        "lon": pano["geometry"]["coordinates"][0],
                        "lat": pano["geometry"]["coordinates"][1],
                        "height": pano["geometry"]["coordinates"][2],
                        "heading": pano["heading"],
                        "roll": pano["roll"],
                        "pitch": pano["pitch"],
                        "thumbnail_url": pano["_links"]["thumbnail"]["href"],
                        "cubic_img_baseurl": pano["cubic_img_baseurl"],
                        "cubic_img_pattern": pano["cubic_img_pattern"],
                        "equirectangular_full": pano["_links"]["equirectangular_full"][
                            "href"
                        ],
                    }
                )

            # Subsequent requests use the "next page" url
            url = data["_links"]["next"]["href"] if data["_links"]["next"] else None
            if url is None:
                break
            response = self.session.get(url)

        return ibis.memtable(panoramas)
