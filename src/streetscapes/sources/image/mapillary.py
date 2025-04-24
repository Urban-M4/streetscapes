# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis
import requests

# --------------------------------------
from environs import Env

# --------------------------------------
import json

# --------------------------------------
from streetscapes.sources import SourceType
from streetscapes.sources.image.base import ImageSourceBase
import pandas as pd

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

    def fetch_image_ids(self, bbox, fields=None, limit=100, extract_latlon=True):
        """
        Fetch Mapillary image IDs within a bounding box.

        See https://www.mapillary.com/developer/api-documentation/#image

        Parameters:
            bbox (list): [west, south, east, north]
            fields (list): List of fields to include in the results. If None, a standard set of fields is returned.
            limit (int): Number of images to request per page (pagination size).
            extract_latlon (bool): Whether to extract latitude and longitude from computed_geometry.

        Returns:
            pd.DataFrame: DataFrame containing image data for the selected fields.
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
        df = pd.DataFrame(all_records)

        # Extract latitude and longitude from computed_geometry if present
        if extract_latlon and "computed_geometry" in df.columns:

            def get_coords(geom):
                # geom is GeoJSON-like dict: {'type':'Point','coordinates':[lon, lat]}
                try:
                    coords = geom.get("coordinates", [None, None])
                    return coords[1], coords[0]
                except Exception:
                    return None, None

            lat_lon = df["computed_geometry"].apply(
                lambda g: pd.Series(get_coords(g), index=["latitude", "longitude"])
            )
            df = pd.concat([df, lat_lon], axis=1)

        return ibis.memtable(df)