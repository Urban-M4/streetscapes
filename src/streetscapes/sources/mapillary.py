# streetscapes/sources/mapillary.py
import logging
import uuid
from pathlib import Path
from time import sleep

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from streetscapes import utils
from streetscapes.project import Project
from streetscapes.utils.bbox import Bbox
from streetscapes.utils.metadata import ImageMeta

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

    @property
    def db_fields(self) -> dict:
        return Project.core_tables["mapillary"]["schema"]  # type: ignore[return-value]

    # NOTE: could make this "fetch_metadata_id" to be similar to bbox retrieval
    def fetch_image_url(self, image_id: str) -> str:
        """Fetch image URL from the Mapillary API by image ID."""
        endpoint = f"https://graph.mapillary.com/{image_id}?fields=thumb_2048_url"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json().get("thumb_2048_url")  # type: ignore[no-any-return]

    def download_image(
        self,
        url: str,
        output_dir: str | Path,
        image_id: int | None,
        uid: uuid.UUID | None = None,
        skip_existing: bool = True,
    ) -> ImageMeta:
        """
        Download image from a URL to output_path.

        Args:
            url: The download URL.
            output_dir: Destination directory.
            image_id: Mapillary image ID.
            uid: Image UUID (from the SHA-256 hash).
            skip_existing: Don't re-download existing images.
        Returns:
            Image metadata.
        """

        output_path = output_dir
        if output_dir is not None:
            output_dir = Path(output_dir)

        content = None
        if uid is not None:
            if output_dir is not None:
                image_path = list(output_dir.glob(f"*{uid}*"))
                if len(image_path) > 0:
                    content = image_path[0].read_bytes()
            if content is None:
                # The image is missing, download it again.
                skip_existing = False

        if uid is None or not skip_existing:
            response = self.session.get(url)
            response.raise_for_status()
            content = response.content

        if content is None:
            raise ValueError(
                f"Failed to download image with UUID '{uid}': empty content"
            )

        meta = utils.get_image_metadata(content)

        if uid is None and output_dir is not None:
            utils.ensure_dir(output_dir)
            output_path = output_dir / f"{meta.uid}.{meta.ext}"
            output_path.write_bytes(meta.content)
            # write mapillary-id -> uuid mapping
            if image_id is not None:
                with (output_dir / str(image_id)).open("w") as f:
                    f.write(str(meta.uid))

        meta.fpath = output_path
        meta.source = "mapillary"

        return meta

    def _fetch_bbox(self, bbox: Bbox, limit: int = 1000) -> list[dict]:
        """Perform the raw API request to Mapillary for a single bounding box tile."""
        logger.debug(f"Fetching metadata for bounding box: {bbox}")

        params = {
            "bbox": ",".join(map(str, bbox)),
            "fields": ",".join(self.db_fields),
            "limit": limit,
        }

        for attempt in range(self.retries):
            try:
                res = self.session.get(self.BASE_URL, params=params, timeout=20)  # type: ignore[arg-type]
                res.raise_for_status()
                return res.json().get("data", [])  # type: ignore[no-any-return]
            except (requests.RequestException, ValueError) as e:
                logger.error(e)
                sleep_time = 0.5 * (attempt + 1)
                logger.info(f"Request failed for {bbox=} - retrying in {sleep_time}")
                sleep(sleep_time)

        logger.warning(f"Failed to retrieve metadata for bounding box: {bbox}")
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
        if df.get("computed_geometry") is None:
            df["computed_geometry"] = None
        else:
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
