"""Extract metadata for images from Mapillary API"""

import json
from time import sleep
from pathlib import Path

import ibis
from environs import Env
import requests
from requests import Session
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from streetscapes import logger
from streetscapes.sources.image.base import ImageSourceBase


def split_bbox(bbox: list[float], tile_size: float) -> list[list[float]]:
    """Split bounding box into tiles

    Args:
        bbox: bounding box [west, south, east, north]
        tile_size: tile size in degrees

    Returns:
        List of bounding box tiles
    """
    tiles = []
    lon = bbox[0]
    while lon < bbox[2]:
        lat = bbox[1]
        while lat < bbox[3]:
            tile = [
                lon,
                lat,
                min(lon + tile_size, bbox[2]),
                min(lat + tile_size, bbox[3]),
            ]
            tiles.append(tile)
            lat += tile_size
        lon += tile_size
    return tiles


def convert_to_ibis(json_records):
    """Convert json records to an ibis table and extract lat/lon if present.

    Args:
        json_records: json of image data to convert

    Returns:
        ibis table of those records
    """
    mt = ibis.memtable(json_records)

    if "computed_geometry" in mt.columns:
        table = mt.mutate(
            lon=mt.computed_geometry.coordinates[0],
            lat=mt.computed_geometry.coordinates[1],
        )
    return table


class Mapillary(ImageSourceBase):
    """
    An interface for downloading and manipulating
    street view images from Mapillary.

    ...

    Attributes:

    base_url: str
        Mapillary url for downloading images
    default_fields: list[str]
        List of metadata fields the API will return
    env:
        An Env object containing loaded configuration options.
    root_dir:
        An optional custom root directory. Defaults to None.

    Methods:
        get_image_url
            Retrieve the URL for an image with the given ID
        create_session
            Create an (authenticated) session for the supplied source
        fetch_image_ids_bbox
            Fetch Mapillary image IDs within a bounding box
        fetch_image_ids_creator
            Fetch Mapillary image IDs by creator username
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
            url="https://graph.mapillary.com",
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
        url = f"{self.url}/{image_id}?fields=thumb_2048_url"

        rq = requests.Request("GET", url, params={"access_token": self.token})
        res = self.session.send(rq.prepare())
        if res.status_code == 200:
            return json.loads(res.content.decode("utf-8"))["thumb_2048_url"]
        else:
            logger.warning(f"Failed to fetch the URL for image {image_id}.")

    def create_session(self) -> requests.Session:
        """
        Create an (authenticated) session for the supplied source.

        Returns:
            A `requests` session.
        """

        session = Session()
        session.headers.update({"Authorization": f"OAuth {self.token}"})
        retries = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def collect_data(self, url, params, filename):
        """Collect metadata records from the API and dump into a json file

        Args:
            url: url for downloading metadata
            params: dictionary of parameters to extract
            filename: filename for the json file

        Returns:
            json of metadata records from API
        """
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Collect data
        records = data.get("data", [])
        with open(filename, "w") as f:
            json.dump(records, f)
        return records

    def fetch_image_ids_bbox(
        self,
        bbox: list[float],
        tile_size: float = 0.01,
        fields: list[str] | None = None,
        limit: int = 1000,
    ):
        """
        Fetch Mapillary image IDs within a bounding box.

        See https://www.mapillary.com/developer/api-documentation/#image

        Parameters:
            bbox: [west, south, east, north]
            tile_size: tile size in degrees, default 0.01 (about 1km)
            fields: List of fields to include in the results. If None, a standard set of fields is returned.
            limit: Number of images to request (Mapillary API limit is 2000, default set to 1000 as 2000 often fails).
            extract_latlon: Whether to extract latitude and longitude from computed_geometry.

        Returns:
            Json containing image data for the selected fields.
        """

        all_records = []

        tiles = split_bbox(bbox, tile_size)

        count = 0
        for tile in tiles:
            count += 1
            if fields is None:
                fields_param = ",".join(self.default_fields)
            else:
                fields_param = ",".join(fields)

            params = {
                "bbox": ",".join(map(str, tile)),
                "fields": fields_param,
                "limit": limit,
            }
            filename = f"image_ids/bbox{count}.json"
            records = collect_data(self.base_url, params, filename)
            all_records.extend(records)

        return all_records

    def fetch_image_ids_creator(
        self,
        creator_username: str,
        fields: list[str] | None = None,
        limit: int = 1000,
    ):
        """
        Fetch Mapillary image IDs by creator username.

        See https://www.mapillary.com/developer/api-documentation/#image

        Parameters:
            creator_username: Username of Mapillary image uploader
            fields: List of fields to include in the results. If None, a standard set of fields is returned.
            limit: Number of images to request per page (pagination size, default 1000).
            extract_latlon: Whether to extract latitude and longitude from computed_geometry.

        Returns:
            Json containing image data for the selected fields.
        """

        if fields is None:
            fields_param = ",".join(self.default_fields)
        else:
            fields_param = ",".join(fields)

        params = {
            "creator_username": creator_username,
            "fields": fields_param,
            "limit": limit,
        }

        url = self.base_url
        all_records = []

        count = 0
        while True:
            count += 1
            filename = f"image_ids/test{count}.json"
            records = collect_data(url, params, filename)
            all_records.extend(records)

            # Check for pagination
            paging = data.get("paging", {})
            print(paging)
            next_url = paging.get("next")
            sleep(1)
            if not next_url:
                break
            # Reset params for next page (next_url already has all params)
            url = next_url
            params = {}

        return all_records
