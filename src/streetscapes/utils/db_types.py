"""Pydantic types for validating source data against the database schema.

Street-level imagery APIs do not always honour their own schema: fields can be
missing, of the wrong type, or contain data belonging to another field. These
annotated types both validate the incoming values and convert them into the
representation expected by the corresponding DuckDB column.
"""

from datetime import UTC, datetime
from typing import Annotated, Any

import orjson as oj
from pydantic import BeforeValidator, Field, StringConstraints
from shapely.geometry import Point


def point_to_wkt(value: Any) -> Any:
    """Convert a GeoJSON point as returned by an API into a WKT string."""
    if not isinstance(value, dict):
        # Anything else (including a WKT string) is left to the field validator.
        return value

    coords = value.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        raise ValueError(f"Malformed point coordinates: {coords!r}")

    try:
        lon, lat = (float(coord) for coord in coords)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Non-numeric point coordinates: {coords!r}") from err

    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"Point coordinates out of range: {coords!r}")

    # WKT facilitates conversion to either GeoPandas or DuckDB geometry.
    return Point(lon, lat).wkt


def epoch_ms_to_datetime(value: Any) -> Any:
    """Convert milliseconds since the (UTC) epoch into an aware datetime."""
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    return value


def as_float_list(value: Any) -> Any:
    """Normalise a scalar or a sequence of numbers into a list of floats."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as err:
            raise ValueError(f"Non-numeric sequence: {value!r}") from err
    return value


def as_json(value: Any) -> Any:
    """Serialise a nested object into a JSON string for a JSON column."""
    if isinstance(value, (dict, list)):
        return oj.dumps(value).decode()
    return value


# WKT point, converted from a GeoJSON object (GEOMETRY column)
WktPoint = Annotated[str, BeforeValidator(point_to_wkt)]

# timezone-aware timestamp, converted from ms since epoch (TIMESTAMPTZ column)
EpochMs = Annotated[datetime, BeforeValidator(epoch_ms_to_datetime)]

# list of floats, tolerating a bare JSON scalar (FLOAT8[] column)
FloatList = Annotated[list[float], BeforeValidator(as_float_list)]

# JSON string, serialised from a nested object (JSON column)
JsonString = Annotated[str, BeforeValidator(as_json)]

# URL, rejecting anything that isn't one (APIs sometimes return garbage)
UrlString = Annotated[str, StringConstraints(pattern=r"^https?://\S+$")]

# non-negative integer, matching UBIGINT columns
UBigInt = Annotated[int, Field(ge=0)]
