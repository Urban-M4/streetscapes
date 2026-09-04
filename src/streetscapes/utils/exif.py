"""EXIF metadata extraction."""

from datetime import UTC, datetime, tzinfo
from functools import cache
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import exifread
import shapely

from streetscapes.utils.logging import logger

if TYPE_CHECKING:
    from pathlib import Path


@cache
def _timezone_finder():
    """Return a cached TimezoneFinder (it loads a sizeable lookup table)."""
    from timezonefinder import TimezoneFinder

    return TimezoneFinder()


def _timezone_at(lon: float, lat: float) -> tzinfo:
    """Guess the timezone a photograph was taken in from its GPS coordinates.

    EXIF capture times are recorded in the camera's local time without any
    indication of the offset, so the coordinates are the only clue available.

    Args:
        lon: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.

    Returns:
        The timezone of the location, defaulting to UTC if it can't be determined.
    """
    zone = _timezone_finder().timezone_at(lng=lon, lat=lat)

    if zone is None:
        logger.warning(
            f"No timezone found for coordinates ({lat}, {lon}). Assuming UTC."
        )
        return UTC

    return ZoneInfo(zone)


def _to_deg(
    dms: list[int | float] | None,
    reference: Literal["N", "E", "S", "W"] | None = None,
) -> float:
    """Convert [deg, min, s] to degrees.

    Args:
        dms: A list containing degrees, minutes and seconds.
        reference: Optional coordinate reference (N, E, S, or W).

    Returns:
        Latitude or longitude coordinate in decimal degrees.
    """
    if dms is None:
        return 0.0

    sign = -1 if reference in ("W", "S") else 1
    deg = dms[0] + dms[1] / 60 + float(dms[2]) / 3600
    return sign * deg


def extract_exif_data(impath: Path) -> dict[str, Any]:
    """Extract EXIF metadata from an image file.

    Args:
        impath: Path to an image.
    """
    with open(impath, "rb") as file_handle:
        tags = exifread.process_file(file_handle)

    # Extract the tags that we are interested in
    lon = tags.get("GPS GPSLongitude")
    if lon is not None:
        lon = lon.values
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if lon_ref is not None:
        lon_ref = lon_ref.values
    lon = _to_deg(lon, lon_ref)

    lat = tags.get("GPS GPSLatitude")
    if lat is not None:
        lat = lat.values
    lat_ref = tags.get("GPS GPSLatitudeRef")
    if lat_ref is not None:
        lat_ref = lat_ref.values
    lat = _to_deg(lat, lat_ref)

    # EXIF timestamps carry no offset, so the location decides how to interpret them.
    tz = _timezone_at(lon, lat)

    mapping = {
        "make": ("Image Make", str),
        "model": ("Image Model", str),
        "orientation": ("Image Orientation", int),
        "timestamp": (
            "Image DateTime",
            lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S").replace(tzinfo=tz),
        ),
        "width": ("EXIF ExifImageWidth", int),
        "height": ("EXIF ExifImageLength", int),
        "altitude": ("GPS GPSAltitude", float),
        "compass_angle": ("GPS GPSTrack", float),
        "geometry": (shapely.Point([lon, lat]), None),
        "is_pano": (None, None),
        "iso": ("EXIF ISOSpeedRatings", int),
        "focal_length": ("EXIF FocalLength", float),
        "exposure": ("EXIF ExposureTime", float),
        "fstop": ("EXIF FNumber", float),
    }

    data = dict.fromkeys(mapping)

    for k, (val, caster) in mapping.items():
        if isinstance(val, str):
            val = tags.get(val)
            if val is not None:
                val = val.values

            if isinstance(val, list):
                val = val[-1]

        if val is None:
            continue

        data[k] = val if caster is None else caster(val)  # type: ignore[operator, assignment]

    return data
