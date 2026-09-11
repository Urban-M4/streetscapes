"""KartaView related functionality."""

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
    FloatList,
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


# Mapping of KartaView API names to more streetscapes-standard names
API_FIELDS = {
    "camera_parameters": "cameraParameters",
    "captured_at": "shotDate",
    "compass_angle": "heading",
    "field_of_view": "fieldOfView",
    "gps_accuracy": "gpsAccuracy",
    "image_url": "fileurlProc",
    "quality_level": "qualityLevel",
    "sequence": "sequenceId",
    "sequence_index": "sequenceIndex",
    "thumb_large_url": "fileurlLTh",
    "thumb_url": "fileurlTh",
    "uploaded_at": "dateAdded",
    "way_id": "wayId",
}


class KartaViewError(RuntimeError):
    """Raised when the KartaView API cannot be reached."""


def _point(lon: Any, lat: Any) -> dict | None:
    """Build a GeoJSON-style point, treating a null island position as missing."""
    try:
        lon, lat = float(lon), float(lat)
    except TypeError, ValueError:
        return {"coordinates": [lon, lat]}

    if lon == 0.0 and lat == 0.0:
        # KartaView reports unmatched photos as (0, 0) rather than as null.
        return None

    return {"coordinates": [lon, lat]}


def _is_pano(projection: Any) -> bool | None:
    """Tell whether a projection is panoramic; KartaView has no flag for it."""
    if projection is None:
        return None
    return str(projection).upper() != "PLANE"


def _timestamp(value: Any) -> Any:
    """Discard the placeholders KartaView uses for a missing timestamp."""
    if isinstance(value, str) and (not value.strip() or value.startswith("0000")):
        return None
    return value


class KartaViewImage(BaseModel):
    """KartaView image record, validated against the database schema.

    KartaView returns every value as a string, spells its fields in camelCase and
    splits positions over separate `lat`/`lng` fields, so raw API records are
    normalised (see `_from_api`) before the fields below are validated.

    Field names and types mirror the `kartaview` table schema, except for `image`
    which is only known after the image itself has been downloaded, and
    `altitude`, which KartaView does not report at all.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    # Required entries
    id: UBigInt
    geometry: WktPoint

    camera_parameters: FloatList | None = None
    captured_at: UtcTimestamp | None = None
    compass_angle: float | None = None
    computed_geometry: WktPoint | None = None
    creator: JsonString | None = None
    field_of_view: float | None = None
    gps_accuracy: float | None = None
    height: UBigInt | None = None
    image_url: UrlString | None = None
    is_pano: bool | None = None
    projection: str | None = None
    quality_level: UBigInt | None = None
    sequence: str | None = None
    sequence_index: UBigInt | None = None
    thumb_large_url: UrlString | None = None
    thumb_url: UrlString | None = None
    uploaded_at: UtcTimestamp | None = None
    way_id: UBigInt | None = None
    width: UBigInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_api(cls, data: Any) -> Any:
        """Map a raw API record onto the fields of this model.

        Values that are already spelled the way this model expects them are kept,
        so an already normalised record passes through unchanged.
        """
        if not isinstance(data, dict):
            return data

        record = dict(data)

        for field, api_field in API_FIELDS.items():
            if api_field in record:
                record.setdefault(field, record[api_field])

        for field in ("captured_at", "uploaded_at"):
            if field in record:
                record[field] = _timestamp(record[field])

        if "lat" in record and "lng" in record:
            record.setdefault("geometry", _point(record["lng"], record["lat"]))

        if "matchLat" in record and "matchLng" in record:
            record.setdefault(
                "computed_geometry", _point(record["matchLng"], record["matchLat"])
            )

        is_pano = _is_pano(record.get("projection"))
        if is_pano is not None:
            record.setdefault("is_pano", is_pano)

        username = record.get("username")
        if username is not None:
            record.setdefault("creator", {"username": username})

        return record

    def to_row(self) -> dict[str, Any]:
        """Get the record as a row for the `kartaview` table."""
        # `image` is filled in once the image has been downloaded, and KartaView
        # reports no altitude.
        return {"image": None, "altitude": None, **self.model_dump()}


def validate_records(records: list[dict]) -> list[KartaViewImage]:
    """Validate raw API records, dropping (and reporting) the invalid ones.

    Args:
        records: Raw image records as returned by the KartaView API.

    Returns:
        The records that passed validation.
    """
    images = []
    for record in records:
        try:
            images.append(KartaViewImage.model_validate(record))
        except ValidationError as err:
            problems = ", ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()
            )
            logger.warning(
                f"Skipping malformed KartaView record (id={record.get('id')!r}):"
                f" {problems}"
            )

    skipped = len(records) - len(images)
    if skipped:
        logger.info(f"Skipped {skipped}/{len(records)} malformed KartaView records.")

    return images


class KartaViewClient:
    """Client for fetching KartaView image metadata via bounding boxes.

    Records are validated against `KartaViewImage` and malformed ones are skipped
    individually. No authentication is required.

    Usage example:
        from streetscapes.sources.kartaview import KartaViewClient

        client = KartaViewClient()

        # bbox: (west, south, east, north)
        bbox = (4.899, 52.372, 4.901, 52.374)

        # Fetch as pandas DataFrame
        df = client.fetch_metadata_bbox(bbox)

        # fetch directly as GeoDataFrame
        gdf = client.fetch_metadata_bbox_gpd(bbox)
    """

    # https://doc.kartaview.org/#section/API-Resources
    LIST_URL = "https://api.openstreetcam.org/1.0/list/nearby-photos/"
    DETAIL_URL = "https://api.openstreetcam.org/2.0/photo/"

    # Maximum number of photos the listing endpoint returns per page.
    PAGE_SIZE = 1000

    # Maximum number of image IDs the photo endpoint accepts per request.
    DETAIL_BATCH_SIZE = 150

    def __init__(self, retries: int = 3, timeout: int = 60):
        """Instantiate the client.

        Args:
            retries : int, optional
                Number of request retries on failure (default is 3).
            timeout : int, optional
                Seconds to wait for a response (default is 60). A request
                normally takes about a second, but the API regularly stalls for
                much longer than that.
        """
        self.session = requests.Session()
        self.retries = retries
        self.timeout = timeout

    @property
    def db_fields(self) -> dict:
        """Get schema's fields."""
        return Project.core_tables["kartaview"]["schema"]  # type: ignore[return-value]

    def fetch_image_url(self, image_id: str | int) -> str | None:
        """Fetch the image URL from the KartaView API by image ID."""
        records = self._fetch_details([str(image_id)])
        if not records:
            return None
        return records[0].get("fileurlProc")  # type: ignore[no-any-return]

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
            image_id: KartaView image ID.
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
            source="kartaview",
            uid=uid,
            skip_existing=skip_existing,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> dict:
        """Perform a request, retrying on failure.

        Returns:
            The parsed response.

        Raises:
            KartaViewError: If every attempt failed.
        """
        for attempt in range(self.retries):
            try:
                res = self.session.request(method, url, timeout=self.timeout, **kwargs)
                res.raise_for_status()
                return res.json()  # type: ignore[no-any-return]
            except (requests.RequestException, ValueError) as e:
                logger.error(e)
                if attempt == self.retries - 1:
                    break
                # The API tends to stall rather than refuse, so back off.
                sleep_time = 2**attempt
                logger.info(f"Request to {url} failed - retrying in {sleep_time}s")
                sleep(sleep_time)

        raise KartaViewError(
            f"The KartaView API at {url} did not respond after {self.retries}"
            " attempts. It is intermittently unavailable, so this is usually"
            " worth retrying."
        )

    def _list_bbox(
        self, bbox: Bbox, limit: int = 1000, pano_only: bool = False
    ) -> list[dict]:
        """List the photos in a bounding box, paging through the results.

        Args:
            bbox: Bounding box as (west, south, east, north).
            limit: Maximum number of photos to list (0 for no limit).
            pano_only: Only list panoramic photos. The API cannot filter on this,
                so the listing is filtered as it comes in, and paging continues
                until `limit` panoramas have been found.

        Returns:
            The (partial) photo records returned by the listing endpoint.
        """
        logger.debug(f"Listing photos for bounding box: {bbox}")

        west, south, east, north = bbox
        # When filtering there is no telling how much of a page is kept, so a
        # small page would only mean many more requests for the same photos.
        if limit <= 0 or pano_only:
            page_size = self.PAGE_SIZE
        else:
            page_size = min(limit, self.PAGE_SIZE)

        photos: list[dict] = []
        page = 1
        while True:
            # The listing endpoint takes a north-west and a south-east corner,
            # each as a 'lat,lon' pair, as multipart form fields.
            payload = {
                "bbTopLeft": f"{north},{west}",
                "bbBottomRight": f"{south},{east}",
                "page": str(page),
                "ipp": str(page_size),
            }
            data = self._request(
                "POST",
                self.LIST_URL,
                files={k: (None, v) for k, v in payload.items()},
            )

            items = data.get("currentPageItems") or []
            if pano_only:
                photos.extend(p for p in items if _is_pano(p.get("projection")))
            else:
                photos.extend(items)

            # A short page is the last one, whatever was kept of it.
            if len(items) < page_size or (0 < limit <= len(photos)):
                break

            page += 1

        # Trim, as limit that isnt a multiple of the page size overshoots on last page
        return photos[:limit] if limit > 0 else photos

    def _fetch_details(self, image_ids: list[str]) -> list[dict]:
        """Fetch full photo records for the given image IDs, in batches."""
        records: list[dict] = []
        for start in range(0, len(image_ids), self.DETAIL_BATCH_SIZE):
            batch = image_ids[start : start + self.DETAIL_BATCH_SIZE]
            data = self._request(
                "GET",
                self.DETAIL_URL,
                params={
                    "id": ",".join(batch),
                    "itemsPerPage": self.DETAIL_BATCH_SIZE,
                },
            )

            records.extend((data.get("result") or {}).get("data") or [])

        return records

    def _fetch_bbox(
        self, bbox: Bbox, limit: int = 1000, pano_only: bool = False
    ) -> list[dict]:
        """Fetch the full photo records for a bounding box.

        Args:
            bbox: Bounding box as (west, south, east, north).
            limit: Maximum number of images to fetch (0 for no limit).
            pano_only: Only fetch panoramic images.

        Returns:
            Raw photo records, enriched with the listing's `username`.
        """
        listed = {
            photo["id"]: photo
            for photo in self._list_bbox(bbox, limit, pano_only=pano_only)
        }

        if not listed:
            return []

        records = self._fetch_details(list(listed))

        # The photo endpoint reports no uploader, so carry it over.
        for record in records:
            username = listed.get(record.get("id"), {}).get("username")
            if username is not None:
                record.setdefault("username", username)

        return records

    def fetch_metadata_bbox(
        self, bbox: Bbox, limit: int = 1000, pano_only: bool = False
    ) -> pd.DataFrame:
        """Fetch metadata for a bounding box and convert to a pandas DataFrame.

        Every record is validated against `KartaViewImage` before being included;
        records that fail validation are skipped and reported.

        Args:
            bbox : tuple[float, float, float, float]
                Bounding box as (west, south, east, north).
            limit : int
                Maximum number of images to fetch (default 1000, 0 for no limit).
            pano_only : bool
                Only fetch panoramic images (default False).

        Returns:
            pd.DataFrame
                DataFrame with KartaView metadata.
        """
        columns = list(self.db_fields)
        images = validate_records(self._fetch_bbox(bbox, limit, pano_only=pano_only))

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
                Maximum number of images to fetch (default 1000, 0 for no limit).
            pano_only : bool
                Only fetch panoramic images (default False).

        Returns:
            gpd.GeoDataFrame
                GeoDataFrame with KartaView metadata and geometry columns.
        """
        df = self.fetch_metadata_bbox(bbox, limit, pano_only)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
        return gdf.set_crs("EPSG:4326")
