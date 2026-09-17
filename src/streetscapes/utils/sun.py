"""Position of the sun."""

import math
from datetime import UTC, datetime

import ephem


def solar_altitude(when: datetime | None, lon: float, lat: float) -> float:
    """Get the altitude of the sun above the horizon, in degrees.

    The altitude is computed with PyEphem, and is the apparent one: it includes
    atmospheric refraction, which lifts the sun by about half a degree at the
    horizon.

    Args:
        when: Moment of observation. A naive datetime is taken to be UTC.
        lon: Longitude of the observer, in degrees.
        lat: Latitude of the observer, in degrees.

    Returns:
        The altitude of the sun, in degrees; NaN if `when` is missing.
    """
    if when is None:
        return math.nan

    if when.tzinfo is not None:
        when = when.astimezone(UTC).replace(tzinfo=None)

    observer = ephem.Observer()
    # PyEphem takes floats as radians (and strings as degrees).
    observer.lon = math.radians(lon)
    observer.lat = math.radians(lat)
    # PyEphem takes naive datetimes as UTC.
    observer.date = when

    return math.degrees(ephem.Sun(observer).alt)
