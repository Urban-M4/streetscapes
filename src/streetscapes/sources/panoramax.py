"""Panoramax related functionality."""

import logging
from time import sleep
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from streetscapes.project import Project
from streetscapes.sources.common import download_image
from streetscapes.utils.db_types import (
    JsonString,
    UBigInt,
    UrlString,
    UtcTimestamp,
    WktPoint,
)

if TYPE_CHECKING:
    import uuid
    from pathlib import Path

    from streetscapes.utils.geo import Bbox
    from streetscapes.utils.metadata import ImageMeta

logger = logging.getLogger(__name__)

# The federated catalogue, which indexes the pictures of every Panoramax
# instance that takes part in it. Pictures stay on the instance that hosts them,
# so their assets are downloaded from there rather than from the catalogue.
# https://docs.panoramax.fr/federated-catalog/
FEDERATED_CATALOGUE = "https://api.panoramax.xyz"

# A picture that covers the full circle is a panorama.
PANORAMIC_FIELD_OF_VIEW = 360

# Mapping of Panoramax STAC property names to more streetscapes-standard names
API_FIELDS = {
    "captured_at": "datetime",
    "compass_angle": "view:azimuth",
    "gps_accuracy": "quality:horizontal_accuracy",
    "license": "license",
    "sequence_index": "geovisio:rank_in_collection",
    "uploaded_at": "created",
}

# Mapping of columns to the STAC asset holding that image
API_ASSETS = {
    "image_url": "hd",
    "thumb_large_url": "sd",
    "thumb_url": "thumb",
}


class PanoramaxError(RuntimeError):
    """Raised when the Panoramax API cannot be reached."""


def _asset_urls(assets: dict | None) -> dict[str, str]:
    """Get the URL of each image size the item offers."""
    assets = assets or {}
    urls = {}
    for field, asset in API_ASSETS.items():
        href = (assets.get(asset) or {}).get("href")
        if href is not None:
            urls[field] = href
    return urls


def _optics(properties: dict) -> dict[str, Any]:
    """Derive the picture's dimensions and its panorama flag from its optics.

    Panoramax reports the dimensions as a pair, and covering the full circle is
    what makes a picture a panorama.
    """
    orientation = properties.get("pers:interior_orientation") or {}
    optics: dict[str, Any] = {}

    dimensions = orientation.get("sensor_array_dimensions")
    if isinstance(dimensions, (list, tuple)) and len(dimensions) == 2:
        optics["width"], optics["height"] = dimensions

    field_of_view = orientation.get("field_of_view")
    if field_of_view is not None:
        optics["field_of_view"] = field_of_view
        optics["is_pano"] = field_of_view == PANORAMIC_FIELD_OF_VIEW

    return optics


def _hosting_instance(links: list | None) -> str | None:
    """Get the instance hosting the picture, which is what its assets point at."""
    for link in links or []:
        if link.get("rel") == "via":
            return link.get("href")  # type: ignore[no-any-return]
    return None


class PanoramaxImage(BaseModel):
    """Panoramax image record, validated against the database schema.

    Panoramax serves STAC items, which nest most of what is needed under
    `properties`, `assets` and `links`, so raw API records are flattened (see
    `_from_api`) before the fields below are validated.

    Field names and types mirror the `panoramax` table schema, except for `image`
    which is only known after the image itself has been downloaded, and
    `altitude`, which Panoramax does not report at all.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    # Required entries
    id: str
    geometry: WktPoint

    captured_at: UtcTimestamp | None = None
    compass_angle: float | None = None
    creator: JsonString | None = None
    field_of_view: float | None = None
    gps_accuracy: float | None = None
    height: UBigInt | None = None
    image_url: UrlString | None = None
    instance: UrlString | None = None
    is_pano: bool | None = None
    license: str | None = None
    sequence: str | None = None
    sequence_index: UBigInt | None = None
    thumb_large_url: UrlString | None = None
    thumb_url: UrlString | None = None
    uploaded_at: UtcTimestamp | None = None
    width: UBigInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_api(cls, data: Any) -> Any:
        """Flatten a STAC item onto the fields of this model.

        Values that are already spelled the way this model expects them are kept,
        so an already flattened record passes through unchanged.
        """
        if not isinstance(data, dict):
            return data

        record = dict(data)
        properties = record.get("properties") or {}

        for field, api_field in API_FIELDS.items():
            if api_field in properties:
                record.setdefault(field, properties[api_field])

        # Panoramax names the uploader with a bare string, but `creator` is a
        # JSON column, holding an object for the other sources too.
        producer = properties.get("geovisio:producer")
        if producer is not None:
            record.setdefault("creator", {"username": producer})

        # The sequence a picture belongs to is its STAC collection.
        if "collection" in record:
            record.setdefault("sequence", record["collection"])

        for field, value in _asset_urls(record.get("assets")).items():
            record.setdefault(field, value)

        for field, value in _optics(properties).items():
            record.setdefault(field, value)

        instance = _hosting_instance(record.get("links"))
        if instance is not None:
            record.setdefault("instance", instance)

        return record

    def to_row(self) -> dict[str, Any]:
        """Get the record as a row for the `panoramax` table."""
        # `image` is filled in once the image has been downloaded, and Panoramax
        # reports no altitude.
        return {"image": None, "altitude": None, **self.model_dump()}


def validate_records(records: list[dict]) -> list[PanoramaxImage]:
    """Validate raw API records, dropping (and reporting) the invalid ones.

    Args:
        records: Raw STAC items as returned by the Panoramax API.

    Returns:
        The records that passed validation.
    """
    images = []
    for record in records:
        try:
            images.append(PanoramaxImage.model_validate(record))
        except ValidationError as err:
            problems = ", ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()
            )
            logger.warning(
                f"Skipping malformed Panoramax record (id={record.get('id')!r}):"
                f" {problems}"
            )

    skipped = len(records) - len(images)
    if skipped:
        logger.info(f"Skipped {skipped}/{len(records)} malformed Panoramax records.")

    return images


class PanoramaxClient:
    """Client for fetching Panoramax image metadata via bounding boxes.

    Queries the federated catalogue by default, which indexes every Panoramax
    instance taking part in the federation, so one query covers them all. Pass a
    single instance to restrict the search to it, or to reach an instance that is
    not federated.

    Records are validated against `PanoramaxImage` and malformed ones are skipped
    individually. No authentication is required.

    Usage example:
        from streetscapes.sources.panoramax import PanoramaxClient

        client = PanoramaxClient()

        # bbox: (west, south, east, north)
        bbox = (4.899, 52.372, 4.901, 52.374)

        # Fetch as pandas DataFrame
        df = client.fetch_metadata_bbox(bbox)

        # fetch directly as GeoDataFrame
        gdf = client.fetch_metadata_bbox_gpd(bbox)

        # a single instance rather than the whole federation
        osm = PanoramaxClient("https://panoramax.openstreetmap.fr")
    """

    # The search endpoint takes an int16 limit and returns no paging cursor, so
    # this is the most pictures a bounding box can yield in one go.
    # https://docs.panoramax.fr/backend/
    MAX_LIMIT = 32767

    def __init__(
        self,
        instance: str = FEDERATED_CATALOGUE,
        retries: int = 3,
        timeout: int = 60,
    ):
        """Instantiate the client.

        Args:
            instance : str, optional
                Base URL of the Panoramax API to query. Defaults to the
                federated catalogue.
            retries : int, optional
                Number of request retries on failure (default is 3).
            timeout : int, optional
                Seconds to wait for a response (default is 60).
        """
        self.instance = instance.rstrip("/")
        self.session = requests.Session()
        self.retries = retries
        self.timeout = timeout

    @property
    def search_url(self) -> str:
        """Get the URL of this instance's search endpoint."""
        return f"{self.instance}/api/search"

    @property
    def db_fields(self) -> dict:
        """Get schema's fields."""
        return Project.core_tables["panoramax"]["schema"]  # type: ignore[return-value]

    def fetch_image_url(self, image_id: str) -> str | None:
        """Fetch the image URL from the Panoramax API by image ID."""
        records = self._request({"ids": image_id, "limit": 1}).get("features") or []
        if not records:
            return None
        return ((records[0].get("assets") or {}).get("hd") or {}).get("href")  # type: ignore[no-any-return]

    def download_image(
        self,
        url: str,
        output_dir: str | Path,
        image_id: int | str | None,
        uid: uuid.UUID | None = None,
        skip_existing: bool = True,
    ) -> ImageMeta:
        """Download image from a URL to output_path.

        Args:
            url: The download URL.
            output_dir: Destination directory.
            image_id: Panoramax image ID.
            uid: Image UUID (from the SHA-256 hash).
            skip_existing: Don't re-download existing images.

        Returns:
            Image metadata.
        """
        return download_image(
            self.session,
            url,
            output_dir,
            image_id,
            source="panoramax",
            uid=uid,
            skip_existing=skip_existing,
        )

    def _request(self, params: dict) -> dict:
        """Perform a search request, retrying on failure.

        Returns:
            The parsed response.

        Raises:
            PanoramaxError: If every attempt failed.
        """
        for attempt in range(self.retries):
            try:
                res = self.session.get(
                    self.search_url, params=params, timeout=self.timeout
                )
                res.raise_for_status()
                return res.json()  # type: ignore[no-any-return]
            except (requests.RequestException, ValueError) as e:
                logger.error(e)
                if attempt == self.retries - 1:
                    break
                sleep_time = 2**attempt
                logger.info(
                    f"Request to {self.search_url} failed - retrying in {sleep_time}s"
                )
                sleep(sleep_time)

        raise PanoramaxError(
            f"The Panoramax API at {self.search_url} did not respond after"
            f" {self.retries} attempts."
        )

    @property
    def filters_pano(self) -> bool:
        """Tell whether the search endpoint can filter on panoramas itself.

        The federated catalogue accepts a filter on the field of view, but single
        instances reject it as unsupported.
        """
        return self.instance == FEDERATED_CATALOGUE

    def _fetch_bbox(
        self, bbox: Bbox, limit: int = 1000, pano_only: bool = False
    ) -> list[dict]:
        """Fetch the STAC items for a bounding box.

        Args:
            bbox: Bounding box as (west, south, east, north).
            limit: Maximum number of images to fetch.
            pano_only: Have the API return panoramic images only, if it can (see
                `filters_pano`). The items are not filtered otherwise.

        Returns:
            Raw STAC items.
        """
        logger.debug(f"Fetching metadata for bounding box: {bbox}")

        capped = limit <= 0 or limit > self.MAX_LIMIT
        if capped:
            # The endpoint rejects a larger limit outright.
            limit = self.MAX_LIMIT

        params: dict[str, Any] = {"bbox": ",".join(map(str, bbox)), "limit": limit}
        if pano_only and self.filters_pano:
            params["filter"] = f"field_of_view={PANORAMIC_FIELD_OF_VIEW}"

        data = self._request(params)
        items = data.get("features") or []

        # Hitting a limit the user chose is expected; only warn when the API cap
        # cut the results short of what was asked for.
        if capped and len(items) == limit:
            logger.warning(
                f"Reached the API cap of {limit} images for bounding box {bbox};"
                " some pictures were left behind. Use smaller tiles to get them."
            )

        return items  # type: ignore[no-any-return]

    def fetch_metadata_bbox(
        self, bbox: Bbox, limit: int = 1000, pano_only: bool = False
    ) -> pd.DataFrame:
        """Fetch metadata for a bounding box and convert to a pandas DataFrame.

        Every record is validated against `PanoramaxImage` before being included;
        records that fail validation are skipped and reported.

        Note:
        ----
        The search endpoint returns no paging cursor and caps the number of
        results at `MAX_LIMIT`, so a bounding box holding more pictures than that
        cannot be covered completely in one call.

        Args:
            bbox : tuple[float, float, float, float]
                Bounding box as (west, south, east, north).
            limit : int
                Maximum number of images to fetch (default 1000).
            pano_only : bool
                Only fetch panoramic images (default False). A single instance
                cannot filter on this itself, so there the limit applies before
                the other images are dropped, and fewer may be returned.

        Returns:
            pd.DataFrame
                DataFrame with Panoramax metadata.
        """
        columns = list(self.db_fields)
        images = validate_records(self._fetch_bbox(bbox, limit, pano_only=pano_only))

        if pano_only:
            # Needed where the API could not filter
            images = [image for image in images if image.is_pano]

        if not images:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame([image.to_row() for image in images], columns=columns)

    def fetch_metadata_bbox_gpd(
        self, bbox: Bbox, limit: int = 1000, pano_only: bool = False
    ) -> gpd.GeoDataFrame:
        """Fetch metadata for a bounding box and convert to a GeoDataFrame.

        Geometry columns are parsed from WKT and the CRS is set to EPSG:4326.

        Args:
            bbox : tuple[float, float, float, float]
                Bounding box as (west, south, east, north).
            limit : int
                Maximum number of images to fetch (default 1000).
            pano_only : bool
                Only fetch panoramic images (default False).

        Returns:
            gpd.GeoDataFrame
                GeoDataFrame with Panoramax metadata and geometry columns.
        """
        df = self.fetch_metadata_bbox(bbox, limit, pano_only)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
        return gdf.set_crs("EPSG:4326")
