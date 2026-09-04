"""EXIF metadata extraction."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import exifread
import shapely

if TYPE_CHECKING:
    from pathlib import Path


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

    mapping = {
        "make": ("Image Make", str),
        "model": ("Image Model", str),
        "orientation": ("Image Orientation", int),
        "timestamp": (
            "Image DateTime",
            lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S"),
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
