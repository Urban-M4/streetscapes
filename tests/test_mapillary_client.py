import threading

import pytest
import requests
from pydantic import ValidationError

from streetscapes.project import Project
from streetscapes.sources.common import concurrent_map
from streetscapes.sources.mapillary import (
    MapillaryClient,
    MapillaryImage,
    validate_records,
)


@pytest.fixture
def valid_record():
    return {
        "id": "1234567890",
        "geometry": {"type": "Point", "coordinates": [4.9, 52.37]},
        "computed_geometry": {"type": "Point", "coordinates": [4.9001, 52.3701]},
        "captured_at": 1672531200000,
        "creator": {"id": "42", "username": "someone"},
        "camera_parameters": [0.85, 0.0, 0.0],
        "camera_type": "perspective",
        "is_pano": False,
        "width": 4032,
        "height": 3024,
        "thumb_2048_url": "https://example.com/image1.jpg",
    }


def test_fetch_metadata_bbox(fake_mapillary_client):
    df = fake_mapillary_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert not df.empty
    assert "geometry" in df.columns
    assert df.iloc[0]["id"] == 1


def test_fetch_metadata_bbox_matches_db_schema(fake_mapillary_client):
    """The DataFrame columns should mirror the `mapillary` table exactly."""
    df = fake_mapillary_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert list(df.columns) == list(Project.core_tables["mapillary"]["schema"])


def test_fetch_metadata_bbox_empty(fake_mapillary_client, monkeypatch):
    """An empty tile still yields a DataFrame with the schema's columns."""
    monkeypatch.setattr(
        type(fake_mapillary_client), "_fetch_bbox", lambda self, bbox, *a, **kw: []
    )
    df = fake_mapillary_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert df.empty
    assert list(df.columns) == list(Project.core_tables["mapillary"]["schema"])


@pytest.mark.parametrize("workers", [1, 8])
def test_fetch_metadata_tiles(fake_mapillary_client, workers):
    """Every tile is fetched, and yields a DataFrame of its own."""
    tiles = [(4.89 + i / 100, 52.37, 4.90 + i / 100, 52.38) for i in range(20)]

    frames = list(fake_mapillary_client.fetch_metadata_tiles(tiles, workers=workers))

    schema = list(Project.core_tables["mapillary"]["schema"])
    assert len(frames) == len(tiles)
    assert all(list(df.columns) == schema for df in frames)


def test_fetch_metadata_tiles_requests_each_tile_once(
    fake_mapillary_client, monkeypatch
):
    requested = []
    lock = threading.Lock()

    def spy(self, bbox, limit=1000, pano_only=False):
        with lock:
            requested.append(bbox)
        return []

    monkeypatch.setattr(type(fake_mapillary_client), "_fetch_bbox", spy)
    tiles = [(4.89 + i / 100, 52.37, 4.90 + i / 100, 52.38) for i in range(20)]

    list(fake_mapillary_client.fetch_metadata_tiles(tiles, workers=8))

    assert sorted(requested) == sorted(tiles)


def test_each_thread_gets_its_own_session():
    """A `requests.Session` is not thread-safe, so threads must not share one."""
    client = MapillaryClient("fake_token")
    # Held on to, so that no session is collected and its id handed to another.
    sessions = list(concurrent_map(lambda _: client.session, range(8), workers=8))

    assert len({id(session) for session in sessions}) > 1
    assert client.session.headers["Authorization"] == "OAuth fake_token"
    # The same thread keeps the same session, so connections are reused.
    assert client.session is client.session


@pytest.mark.parametrize("pano_only", [False, True])
def test_fetch_filters_panoramas_in_the_api(monkeypatch, pano_only):
    """The API filters on `is_pano`, so the limit counts panoramas only."""
    seen: dict = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    def spy(self, url, params, **kwargs):
        seen.update(params)
        return _Response()

    monkeypatch.setattr(requests.Session, "get", spy)

    MapillaryClient("fake_token").fetch_metadata_bbox(
        (4.89, 52.37, 4.91, 52.38), pano_only=pano_only
    )

    assert seen.get("is_pano") == ("true" if pano_only else None)


def test_fetch_daytime_only(fake_mapillary_client, monkeypatch, valid_record):
    """Images captured after dark, or at an unknown time, are dropped."""
    day = {**valid_record, "id": 1, "captured_at": 1736942400000}  # 2025-01-15 12:00Z
    night = {**valid_record, "id": 2, "captured_at": 1736962200000}  # 17:30Z
    unknown = {**valid_record, "id": 3, "captured_at": None}
    monkeypatch.setattr(
        type(fake_mapillary_client),
        "_fetch_bbox",
        lambda self, bbox, *a, **kw: [day, night, unknown],
    )
    bbox = (4.89, 52.37, 4.91, 52.38)

    df = fake_mapillary_client.fetch_metadata_bbox(bbox, daytime_only=True)

    assert list(df["id"]) == [1]
    assert len(fake_mapillary_client.fetch_metadata_bbox(bbox)) == 3


def test_model_matches_db_schema():
    """The model and the `mapillary` table must not drift apart."""
    schema = Project.core_tables["mapillary"]["schema"]

    # `image` is not part of the API response, so it isn't a model field.
    assert set(MapillaryImage.model_fields) | {"image"} == set(schema)


def test_model_conversions(valid_record):
    image = MapillaryImage.model_validate(valid_record)

    assert image.id == 1234567890
    assert image.geometry == "POINT (4.9 52.37)"
    assert image.captured_at.isoformat() == "2023-01-01T00:00:00+00:00"
    # The `creator` column is JSON, so nested objects are serialised.
    assert image.creator == '{"id":"42","username":"someone"}'
    assert image.to_row()["image"] is None


def test_model_defaults_missing_fields():
    """Fields the API leaves out become NULLs rather than missing columns."""
    image = MapillaryImage.model_validate(
        {"id": 1, "geometry": {"coordinates": [4.9, 52.37]}}
    )

    assert image.thumb_2048_url is None
    assert image.computed_geometry is None
    assert set(image.to_row()) == set(Project.core_tables["mapillary"]["schema"])


def test_model_scalar_camera_parameters(valid_record):
    """A bare scalar is accepted for the list-valued columns."""
    image = MapillaryImage.model_validate({**valid_record, "camera_parameters": 0.85})

    assert image.camera_parameters == [0.85]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # The API sometimes puts an image URL in the ID field.
        ("id", "https://example.com/image1.jpg"),
        ("id", -1),
        ("geometry", {"type": "Point", "coordinates": [4.9, 952.37]}),
        ("geometry", {"type": "Point", "coordinates": [4.9]}),
        ("geometry", {"type": "Point", "coordinates": ["north", "east"]}),
        ("geometry", None),
        ("thumb_2048_url", "12345"),
        ("width", "wide"),
        ("camera_parameters", ["a", "b"]),
        ("camera_type", {"type": "perspective"}),
    ],
)
def test_model_rejects_bad_fields(valid_record, field, value):
    with pytest.raises(ValidationError):
        MapillaryImage.model_validate({**valid_record, field: value})


def test_model_requires_id_and_geometry(valid_record):
    for field in ("id", "geometry"):
        record = {k: v for k, v in valid_record.items() if k != field}
        with pytest.raises(ValidationError):
            MapillaryImage.model_validate(record)


def test_validate_records_skips_bad_images(valid_record):
    """A malformed image is dropped without discarding the rest of the tile."""
    records = [
        valid_record,
        {**valid_record, "id": "https://example.com/image2.jpg"},
        {**valid_record, "id": 2, "geometry": {"coordinates": [4.9, 952.37]}},
        {**valid_record, "id": 3},
    ]

    images = validate_records(records)

    assert [image.id for image in images] == [1234567890, 3]
