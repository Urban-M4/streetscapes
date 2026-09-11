import pytest
from pydantic import ValidationError

from streetscapes.project import Project
from streetscapes.sources.panoramax import (
    FEDERATED_CATALOGUE,
    PanoramaxClient,
    PanoramaxImage,
    validate_records,
)


def test_defaults_to_the_federated_catalogue():
    """One query should cover every federated instance unless told otherwise."""
    assert PanoramaxClient().instance == FEDERATED_CATALOGUE
    assert PanoramaxClient().search_url == f"{FEDERATED_CATALOGUE}/api/search"


def test_instance_override():
    """A single instance can be queried instead, however it is spelled."""
    client = PanoramaxClient("https://panoramax.openstreetmap.fr/")

    assert client.search_url == "https://panoramax.openstreetmap.fr/api/search"


def test_fetch_metadata_bbox(fake_panoramax_client):
    df = fake_panoramax_client.fetch_metadata_bbox((2.34, 48.85, 2.35, 48.86))

    assert not df.empty
    assert "geometry" in df.columns
    assert df.iloc[0]["id"] == "2ad2b303-59f5-4f8b-a7c1-ccfaef465fbe"


def test_fetch_metadata_bbox_matches_db_schema(fake_panoramax_client):
    """The DataFrame columns should mirror the `panoramax` table exactly."""
    df = fake_panoramax_client.fetch_metadata_bbox((2.34, 48.85, 2.35, 48.86))

    assert list(df.columns) == list(Project.core_tables["panoramax"]["schema"])


def test_fetch_metadata_bbox_empty(fake_panoramax_client, monkeypatch):
    """An empty tile still yields a DataFrame with the schema's columns."""
    monkeypatch.setattr(
        type(fake_panoramax_client), "_request", lambda self, params: {"features": []}
    )
    df = fake_panoramax_client.fetch_metadata_bbox((2.34, 48.85, 2.35, 48.86))

    assert df.empty
    assert list(df.columns) == list(Project.core_tables["panoramax"]["schema"])


@pytest.fixture
def spy_request(fake_panoramax_client, monkeypatch):
    """Capture the parameters the client sends to the search endpoint."""
    seen: dict = {}

    def spy(self, params):
        seen.update(params)
        return {"features": []}

    monkeypatch.setattr(type(fake_panoramax_client), "_request", spy)
    return seen


@pytest.mark.parametrize("limit", [99999, 0])
def test_fetch_caps_limit_at_the_api_maximum(fake_panoramax_client, spy_request, limit):
    """The endpoint rejects a limit above its own maximum outright."""
    fake_panoramax_client.fetch_metadata_bbox((2.34, 48.85, 2.35, 48.86), limit=limit)

    assert spy_request["limit"] == PanoramaxClient.MAX_LIMIT


def test_fetch_queries_the_bbox_as_given(fake_panoramax_client, spy_request):
    """The bounding box must not be widened; the limit applies to what it covers.

    Snapping it to a coarse grid would spend the limit on pictures outside the
    area that was asked for, and return none of the ones inside it.
    """
    fake_panoramax_client.fetch_metadata_bbox(
        (2.336698, 48.865696, 2.346236, 48.870650)
    )

    assert spy_request["bbox"] == "2.336698,48.865696,2.346236,48.87065"


def test_pano_only_filters_in_the_federated_catalogue(
    fake_panoramax_client, spy_request
):
    """The catalogue filters on the field of view, so the limit counts panoramas."""
    fake_panoramax_client.fetch_metadata_bbox(
        (2.34, 48.85, 2.35, 48.86), pano_only=True
    )

    assert spy_request["filter"] == "field_of_view=360"


@pytest.mark.parametrize(
    ("instance", "pano_only"),
    [
        (FEDERATED_CATALOGUE, False),
        # A single instance rejects the filter as unsupported.
        ("https://panoramax.openstreetmap.fr", True),
    ],
)
def test_pano_only_sends_no_filter(spy_request, instance, pano_only):
    PanoramaxClient(instance).fetch_metadata_bbox(
        (2.34, 48.85, 2.35, 48.86), pano_only=pano_only
    )

    assert "filter" not in spy_request


@pytest.mark.parametrize("instance", [FEDERATED_CATALOGUE, "https://example.com"])
def test_pano_only_drops_narrow_pictures(panoramax_item, monkeypatch, instance):
    """Whether or not the API could filter, only panoramas are returned."""
    narrow = {
        **panoramax_item,
        "id": "narrow",
        "properties": {
            **panoramax_item["properties"],
            "pers:interior_orientation": {"field_of_view": 95},
        },
    }
    unknown = {**panoramax_item, "id": "unknown", "properties": {}}
    monkeypatch.setattr(
        PanoramaxClient,
        "_request",
        lambda self, params: {"features": [panoramax_item, narrow, unknown]},
    )
    client = PanoramaxClient(instance)
    bbox = (2.34, 48.85, 2.35, 48.86)

    assert list(client.fetch_metadata_bbox(bbox, pano_only=True)["id"]) == [
        panoramax_item["id"]
    ]
    assert len(client.fetch_metadata_bbox(bbox)) == 3


def test_model_matches_db_schema():
    """The model and the `panoramax` table must not drift apart."""
    schema = Project.core_tables["panoramax"]["schema"]

    # `image` is not part of the API response and Panoramax reports no altitude,
    # so neither is a model field.
    assert set(PanoramaxImage.model_fields) | {"image", "altitude"} == set(schema)


def test_model_conversions(panoramax_item):
    image = PanoramaxImage.model_validate(panoramax_item)

    assert image.id == "2ad2b303-59f5-4f8b-a7c1-ccfaef465fbe"
    assert image.geometry == "POINT (2.3425668 48.8582587)"
    assert image.captured_at.isoformat() == "2024-10-05T14:24:57+00:00"
    assert image.uploaded_at.isoformat() == "2024-10-05T18:24:28.811997+00:00"
    assert image.compass_angle == 280
    assert image.gps_accuracy == 4.0
    assert image.license == "etalab-2.0"
    # The sequence a picture belongs to is its STAC collection.
    assert image.sequence == "9802529e-5ec8-433a-95ac-e9d37bf3be7e"
    assert image.sequence_index == 344


def test_model_wraps_the_bare_producer(panoramax_item):
    """`creator` is a JSON column, so a bare producer name has to be wrapped."""
    image = PanoramaxImage.model_validate(panoramax_item)

    assert image.creator == '{"username":"tdelmas"}'


def test_model_flattens_assets(panoramax_item):
    image = PanoramaxImage.model_validate(panoramax_item)

    assert image.image_url == "https://example.com/pictures/hd.jpg"
    assert image.thumb_large_url == "https://example.com/pictures/sd.jpg"
    assert image.thumb_url == "https://example.com/pictures/thumb.jpg"


def test_model_records_the_hosting_instance(panoramax_item):
    """Assets live on the instance that hosts them, not on the catalogue."""
    image = PanoramaxImage.model_validate(panoramax_item)

    assert image.instance == "https://panoramax.ign.fr"


def test_model_derives_dimensions_and_pano(panoramax_item):
    """Panoramax reports the dimensions as a pair and no panorama flag."""
    pano = PanoramaxImage.model_validate(panoramax_item)

    assert (pano.width, pano.height) == (5760, 2880)
    assert (pano.field_of_view, pano.is_pano) == (360, True)


@pytest.mark.parametrize("field_of_view", [14, 40, 92, 95])
def test_model_narrow_field_of_view_is_not_panoramic(panoramax_item, field_of_view):
    """Only a picture covering the full circle is a panorama."""
    item = {**panoramax_item, "properties": {**panoramax_item["properties"]}}
    item["properties"]["pers:interior_orientation"] = {
        "field_of_view": field_of_view,
        "sensor_array_dimensions": [4000, 3000],
    }
    image = PanoramaxImage.model_validate(item)

    assert image.is_pano is False
    assert image.field_of_view == field_of_view


def test_model_defaults_missing_fields():
    """Fields the API leaves out become NULLs rather than missing columns."""
    image = PanoramaxImage.model_validate(
        {"id": "abc", "geometry": {"coordinates": [2.34, 48.85]}}
    )

    assert image.image_url is None
    assert image.is_pano is None
    assert image.instance is None
    assert set(image.to_row()) == set(Project.core_tables["panoramax"]["schema"])
    # Panoramax reports no altitude, but the column is part of the schema.
    assert image.to_row()["altitude"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("geometry", {"type": "Point", "coordinates": [2.34, 948.85]}),
        ("geometry", {"type": "Point", "coordinates": [2.34]}),
        ("geometry", {"type": "Point", "coordinates": ["east", "north"]}),
        ("geometry", None),
    ],
)
def test_model_rejects_bad_geometry(panoramax_item, field, value):
    with pytest.raises(ValidationError):
        PanoramaxImage.model_validate({**panoramax_item, field: value})


def test_model_rejects_bad_asset_url(panoramax_item):
    item = {**panoramax_item, "assets": {"hd": {"href": "not-a-url"}}}

    with pytest.raises(ValidationError):
        PanoramaxImage.model_validate(item)


def test_model_requires_id_and_geometry(panoramax_item):
    for field in ("id", "geometry"):
        record = {k: v for k, v in panoramax_item.items() if k != field}
        with pytest.raises(ValidationError):
            PanoramaxImage.model_validate(record)


def test_validate_records_skips_bad_images(panoramax_item):
    """A malformed item is dropped without discarding the rest of the tile."""
    records = [
        panoramax_item,
        {**panoramax_item, "id": "bad", "geometry": {"coordinates": [2.34, 948.85]}},
        {**panoramax_item, "id": "good"},
    ]

    images = validate_records(records)

    assert [image.id for image in images] == [panoramax_item["id"], "good"]
