# --------------------------------------
import json

# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from environs import Env

# --------------------------------------
import requests

# --------------------------------------
from streetscapes import logger
from streetscapes.sources.image.base import ImageSourceBase


class Mapillary(ImageSourceBase):
    """TODO: Add docstrings"""

    def __init__(
        self,
        env: Env = None,
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
            return json.loads(res.content.decode("utf-8"))[f"thumb_original_url"]

        logger.warning(f"Failed to fetch the URL for image {image_id}.")

    def create_session(self) -> requests.Session:
        """
        Create an (authenticated) session for the supplied source.

        Returns:
            A `requests` session.
        """

        session = requests.Session()
        session.headers.update({"Authorization": f"OAuth {self.token}"})
        return session

    def fetch_image_ids(
        self,
        bbox: list[float],
        fields: list[str] | None = None,
        limit: int = 500,
        extract_latlon: bool = True,
    ):
        """
        Fetch Mapillary image IDs within a bounding box.

        See https://www.mapillary.com/developer/api-documentation/#image

        Parameters:
            bbox: [west, south, east, north]
            fields: List of fields to include in the results. If None, a standard set of fields is returned.
            limit: Number of images to request per page (pagination size).
            extract_latlon: Whether to extract latitude and longitude from computed_geometry.

        Returns:
            Ibis table containing image data for the selected fields.
        """
        base_url = "https://graph.mapillary.com/images"
        default_fields = [
            "id",
            "altitude",
            "atomic_scale",
            # "camera_parameters",
            "camera_type",
            "captured_at",
            "compass_angle",
            "computed_altitude",
            "computed_compass_angle",
            "computed_geometry",
            "computed_rotation",
            # "creator",
            "exif_orientation",
            "geometry",
            "height",
            "is_pano",
            "make",
            "model",
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
            # "merge_cc",
            # "mesh",
            "sequence",
            # "sfm_cluster",
            "width",
            # "detections",
        ]
        if fields is None:
            fields_param = ",".join(default_fields)
        else:
            fields_param = ",".join(fields)

        params = {
            "bbox": ",".join(map(str, bbox)),
            "fields": fields_param,
            "limit": limit,
        }

        all_records = []
        url = base_url

        while True:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            # Collect data
            records = data.get("data", [])
            all_records.extend(records)

            # Check for pagination
            paging = data.get("paging", {})
            next_url = paging.get("next")
            if not next_url:
                break
            # Reset params for next page (next_url already has all params)
            url = next_url
            params = {}

        # Convert to Dataframe
        mt = ibis.memtable(all_records)

        # Extract latitude and longitude from computed_geometry if present
        if extract_latlon and "computed_geometry" in mt.columns:

            mt = mt.mutate(
                lon=mt.computed_geometry.coordinates[0],
                lat=mt.computed_geometry.coordinates[1],
            )

        return mt
