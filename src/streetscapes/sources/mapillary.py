# streetscapes/sources/mapillary.py
import logging
from pathlib import Path
from time import sleep

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from streetscapes.utils.bbox import Bbox

logger = logging.getLogger(__name__)



class MapillaryClient:
    """Minimal client for fetching Mapillary image metadata via bounding boxes.

    Handles authentication, retries, and conversion of geometry fields to WKT
    strings suitable for use with GeoPandas or DuckDB.

    Usage example:
        import os
        from dotenv import load_dotenv
        from streetscapes.sources.mapillary import MapillaryClient

        load_dotenv()
        token = os.getenv("MAPILLARY_TOKEN")
        client = MapillaryClient(token)

        # bbox: (west, south, east, north)
        bbox = (4.899, 52.372, 4.901, 52.374)

        # Fetch as pandas DataFrame
        df = client.fetch_metadata_bbox(bbox)

        # Or fetch directly as GeoDataFrame
        gdf = client.fetch_metadata_bbox_gpd(bbox)

    Methods
    -------
    fetch_metadata_bbox(bbox: tuple[float, float, float, float], limit: int = 1000) -> pd.DataFrame
        Fetch metadata for a bounding box and return as a pandas DataFrame.
    fetch_metadata_bbox_gpd(bbox: tuple[float, float, float, float], limit: int = 1000) -> gpd.GeoDataFrame
        Fetch metadata for a bounding box and return as a GeoDataFrame with CRS EPSG:4326.

    """

    BASE_URL = "https://graph.mapillary.com/images"
    # https://www.mapillary.com/developer/api-documentation#image
    DEFAULT_FIELDS = [
        "altitude",
        "atomic_scale",
        "camera_type",
        "captured_at",
        "compass_angle",
        "computed_altitude",
        "computed_compass_angle",
        "computed_geometry",
        "computed_rotation",
        "creator",
        "exif_orientation",
        "geometry",
        "height",
        "id",
        "is_pano",
        "make",
        "model",
        "sequence",
        "sequence",
        "thumb_1024_url",
        "thumb_2048_url",
        "thumb_256_url",
        "thumb_original_url",
        "width",
        "camera_parameters",
        # "detections",
        # "merge_cc",
        # "mesh",
        # "sfm_cluster",
    ]

    def __init__(self, token: str, retries: int = 3):
        """Instantiate the client.

        Parameters
        ----------
        token : str
            Mapillary OAuth token.
        retries : int, optional
            Number of request retries on failure (default is 3).

        """
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {token}"})
        self.retries = retries

    # NOTE: could make this "fetch_metadata_id" to be similar to bbox retrieval
    def fetch_image_url(self, image_id: str) -> str:
        """Fetch image URL from the Mapillary API by image ID."""
        endpoint = f"https://graph.mapillary.com/{image_id}?fields=thumb_2048_url"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json().get("thumb_2048_url")

    def download_image(self, url: str, output_path: Path) -> Path:
        """Download image from a URL to output_path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = self.session.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)

    def _fetch_bbox(self, bbox: Bbox, limit: int = 1000) -> list[dict]:
        """Perform the raw API request to Mapillary for a single bounding box tile."""
        logger.debug(f"Fetching metadata for bounding box: {bbox}")

        params = {
            "bbox": ",".join(map(str, bbox)),
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": limit,
        }
        for attempt in range(self.retries):
            try:
                res = self.session.get(self.BASE_URL, params=params, timeout=10)
                res.raise_for_status()
                return res.json().get("data", [])
            except (requests.RequestException, ValueError):
                sleep_time = 0.5 * (attempt + 1)
                logger.info(f"Request failed for {bbox=} - retrying in {sleep_time}")
                sleep(sleep_time)

        logger.warning(f"Failed to retrieve metadata for bbounding box: {bbox}")
        return []

    def fetch_metadata_bbox(self, bbox: Bbox, limit: int = 1000) -> pd.DataFrame:
        """Fetch metadata for a bounding box and convert to a pandas DataFrame.

        Geometry columns are converted to WKT strings for downstream processing
        with GeoPandas or spatial databases like DuckDB.

        Note
        ----
        The Mapillary API endpoint doesn't support pagination beyond ~2000 results.
        For dense areas (like Amsterdam), consider splitting your bounding box into
        smaller tiles (~0.001 deg) to ensure complete coverage.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as (west, south, east, north).
        limit : int
            Maximum number of images to fetch (default 1000).

        Returns
        -------
        pd.DataFrame
            DataFrame with Mapillary metadata and WKT geometry columns.

        """
        records = self._fetch_bbox(bbox, limit)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Geometry colums are dicts, flatten to wkt strings instead
        def unpack_geometry(geometry):
            if isinstance(geometry, dict) and "coordinates" in geometry:
                # wkt facilitates conversion to either geopandas or duckdb geometry
                return Point(geometry["coordinates"]).wkt
            return None

        df["geometry"] = df["geometry"].apply(unpack_geometry)
        df["computed_geometry"] = df["computed_geometry"].apply(unpack_geometry)

        return df

    def fetch_metadata_bbox_gpd(
        self, bbox: Bbox, limit: int = 1000
    ) -> gpd.GeoDataFrame:
        """Fetch metadata for a bounding box and convert to a GeoDataFrame.

        Geometry columns are parsed from WKT and the CRS is set to EPSG:4326.

        Note
        ----
        The Mapillary API endpoint doesn't support pagination beyond ~2000 results.
        For dense areas (like Amsterdam), consider splitting your bounding box into
        smaller tiles (~0.001 deg) to ensure complete coverage.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as (west, south, east, north).
        limit : int
            Maximum number of images to fetch (default 1000).

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with Mapillary metadata and geometry columns.

        """
        df = self.fetch_metadata_bbox(bbox, limit)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
        return gdf.set_crs("EPSG:4326")
