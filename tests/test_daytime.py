import math
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from streetscapes.sources import common
from streetscapes.sources.common import (
    MIN_SUN_ALTITUDE,
    captured_in_daylight,
    keep_daytime,
)
from streetscapes.utils.sun import solar_altitude

AMSTERDAM = (4.9, 52.37)


def test_solar_altitude_summer_noon():
    # Solar noon in Amsterdam at the summer solstice: 90 - 52.37 + 23.44
    altitude = solar_altitude(datetime(2025, 6, 21, 11, 40, tzinfo=UTC), *AMSTERDAM)

    assert altitude == pytest.approx(61.07, abs=0.1)


def test_solar_altitude_winter_evening():
    """A January image at half past five (local time) is taken after sunset."""
    when = datetime(2025, 1, 15, 17, 30, tzinfo=timezone(timedelta(hours=1)))

    assert solar_altitude(when, *AMSTERDAM) < 0


def test_solar_altitude_midnight_sun():
    """North of the arctic circle, the sun stays up at midnight in summer."""
    tromso = (18.96, 69.65)

    assert solar_altitude(datetime(2025, 6, 21, 23, 0, tzinfo=UTC), *tromso) > 0


@pytest.mark.parametrize(("lon", "expected"), [(0, 90), (180, -90)])
def test_solar_altitude_longitude(lon, expected):
    """At the equinox, noon UTC puts the sun overhead at 0°E and below at 180°E."""
    altitude = solar_altitude(datetime(2025, 3, 20, 12, 7, tzinfo=UTC), lon, 0)

    assert altitude == pytest.approx(expected, abs=1)


def test_solar_altitude_time_zones():
    """The same moment gives the same altitude, whatever its zone; naive is UTC."""
    utc = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    cet = utc.astimezone(timezone(timedelta(hours=1)))
    naive = utc.replace(tzinfo=None)

    altitudes = {solar_altitude(when, *AMSTERDAM) for when in (utc, cet, naive)}

    assert len(altitudes) == 1


def test_solar_altitude_missing_time():
    assert math.isnan(solar_altitude(None, *AMSTERDAM))


def _image(captured_at, lon=AMSTERDAM[0], lat=AMSTERDAM[1]):
    return SimpleNamespace(captured_at=captured_at, geometry=f"POINT ({lon} {lat})")


def test_keep_daytime():
    day = _image(datetime(2025, 1, 15, 12, 0, tzinfo=UTC))
    night = _image(datetime(2025, 1, 15, 17, 30, tzinfo=UTC))
    unknown = _image(None)

    assert keep_daytime([day, night, unknown]) == [day]
    assert keep_daytime([]) == []


@pytest.mark.parametrize(
    ("altitude", "daytime"),
    [(MIN_SUN_ALTITUDE, True), (MIN_SUN_ALTITUDE - 0.01, False), (45, True)],
)
def test_captured_in_daylight_threshold(monkeypatch, altitude, daytime):
    """The sun has to be at least `MIN_SUN_ALTITUDE` high."""
    monkeypatch.setattr(common, "solar_altitude", lambda *args: altitude)

    assert captured_in_daylight([_image(datetime(2025, 1, 15, tzinfo=UTC))]) == [
        daytime
    ]


def test_captured_in_daylight_uses_the_position(monkeypatch):
    """The geometry's x is the longitude, and its y the latitude."""
    seen = []
    monkeypatch.setattr(
        common, "solar_altitude", lambda when, lon, lat: seen.append((lon, lat)) or 0
    )

    captured_in_daylight([_image(datetime(2025, 1, 15, tzinfo=UTC), 4.9, 52.37)])

    assert seen == [(4.9, 52.37)]
