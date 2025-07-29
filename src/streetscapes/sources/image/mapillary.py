"""Extract metadata for images from Mapillary API"""

import json
import glob
from time import sleep
from pathlib import Path

import ibis
from environs import Env
import requests
from requests import Session
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import pandas as pd
from pandas import json_normalize
import geopandas as gpd

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
    # TODO: Would be nice to snap to a raster
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
        fetch_metadata_bbox
            Fetch Mapillary image IDs within a bounding box
        fetch_metadata_creator
            Fetch Mapillary image IDs by creator username
        convert_to_ibis:
            Convert records to an ibis table
        convert_to_gdf:
            Convert json records to a GeoDataFrame
        json_to_gdf:
            Load a json file with Mapillary API metadata and convert it to a GeoDataFrame
        concat_metadata:
            Concatenate multiple GeoDataFrames from json files containing Mapillary API data into a single GeoDataFrame
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
            env: Env object containing loaded configuration options.
            root_dir: optional custom root directory. Defaults to None.
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
            image_id: the image ID.

        Returns:
            str: the URL to query.
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
        return data

    def fetch_metadata_bbox(
        self,
        bbox: list[float],
        tile_size: float = 0.01,
        fields: list[str] | None = None,
        limit: int = 1000,
        bbox_name: str = "bbox",
        overwrite: bool = False,
    ):
        """
        Fetch Mapillary image IDs within a bounding box.

        See https://www.mapillary.com/developer/api-documentation/#image

        Args:
            bbox: [west, south, east, north]
            tile_size: tile size in degrees, default 0.01 (about 1km)
            fields: List of fields to include in the results. If None, a standard set of fields is returned.
            limit: Number of images to request (Mapillary API limit is 2000, default set to 1000 as 2000 often fails).
            extract_latlon: Whether to extract latitude and longitude from computed_geometry.
            bbox_name: bounding box name for the file pattern

        Returns:
            Json containing image data for the selected fields.
        """

        metadata_dir = Path(f"{self.root_dir}/metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        all_records = []

        tiles = split_bbox(bbox, tile_size)

        for tile in tiles:
            rounded_tile = [round(v, 2) for v in tile]
            tile_str = "_".join(map(str, rounded_tile))
            filename = Path(metadata_dir, f"{bbox_name}{tile_str}.json")

            if not filename.is_file() or overwrite:
                if fields is None:
                    fields_param = ",".join(self.default_fields)
                else:
                    fields_param = ",".join(fields)

                params = {
                    "bbox": ",".join(map(str, tile)),
                    "fields": fields_param,
                    "limit": limit,
                }
                data = self.collect_data(self.base_url, params, filename)
                records = data.get("data", [])
                all_records.extend(records)
            else:
                print(f"{filename} already exists, skip.")

        return all_records

    def fetch_metadata_creator(
        self,
        creator_username: str,
        fields: list[str] | None = None,
        limit: int = 1000,
    ):
        """
        Fetch Mapillary image IDs by creator username.

        See https://www.mapillary.com/developer/api-documentation/#image

        Args:
            creator_username: Username of Mapillary image uploader
            fields: List of fields to include in the results. If None, a standard set of fields is returned.
            limit: Number of images to request per page (pagination size, default 1000).
            extract_latlon: Whether to extract latitude and longitude from computed_geometry.

        Returns:
            Json containing image data for the selected fields.
        """
        # TODO: Include a check for metadata already existing

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
        metadata_dir = Path(f"{self.root_dir}/metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        all_records = []

        count = 0
        while True:
            count += 1
            filename = Path(metadata_dir, f"{creator_username}{count}.json")
            data = self.collect_data(url, params, filename)
            records = data.get("data", [])
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

    def convert_to_ibis(self, json_records):
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

    def convert_to_gdf(self, dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Add lon and lat to the dataframe and convert to a geopandas GeoDataFrame

        The dataframe must have a column named computed_geometry.coordinates which is a list of two floats.
        The crs is set to EPSG:4326.

        Args:
            dataframe: pandas DataFrame with Mapillary API metadata.

        Returns:
            geopandas GeoDataFrame with geometry column.
        """
        if "computed_geometry.coordinates" in dataframe.columns and not isinstance(
            dataframe["computed_geometry.coordinates"][0], float
        ):
            dataframe["lon"] = [x[0] for x in dataframe["geometry.coordinates"]]
            dataframe["lat"] = [x[1] for x in dataframe["geometry.coordinates"]]
            gdf = gpd.GeoDataFrame(
                dataframe, geometry=gpd.points_from_xy(dataframe.lon, dataframe.lat)
            )
            gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
            return gdf

    def json_to_gdf(self, json_file: Path | str) -> gpd.GeoDataFrame:
        """
        Load a json file with Mapillary API metadata and convert it to a GeoDataFrame.

        Args:
            json_file: Path to the json file to load.

        Returns:
            GeoDataFrame with geometry column.
        """
        with open(json_file, "r") as file:
            data = json.load(file)
        norm_df = json_normalize(data)
        gdf = self.convert_to_gdf(norm_df)
        return gdf

    def concat_metadata(self, metadata_path: Path | str) -> gpd.GeoDataFrame:
        """
        Concatenate multiple GeoDataFrames from json files containing Mapillary API data into a single GeoDataFrame.

        Args:
            metadata_path: path to the json files to concatenate.

        Returns:
            GeoDataFrame with all the data from the concatenated files.
        """
        metadata = glob.glob(metadata_path)
        gdfs = []
        for f in metadata:
            gdf = self.json_to_gdf(f)
            gdfs.append(gdf)
        return pd.concat(gdfs)
