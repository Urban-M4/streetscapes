import pytest
from pydantic import ValidationError

from streetscapes.project import Project
from streetscapes.sources.mapillary import MapillaryImage, validate_records


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
        type(fake_mapillary_client), "_fetch_bbox", lambda self, bbox, limit=1000: []
    )
    df = fake_mapillary_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert df.empty
    assert list(df.columns) == list(Project.core_tables["mapillary"]["schema"])


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
