import pytest
import requests
from pydantic import ValidationError

from streetscapes.project import Project
from streetscapes.sources.kartaview import (
    KartaViewClient,
    KartaViewError,
    KartaViewImage,
    validate_records,
)


@pytest.fixture
def valid_record():
    """A photo record as returned by the KartaView photo endpoint.

    Every value is a string, as KartaView reports them.
    """
    return {
        "id": "1234567890",
        "lat": "52.37",
        "lng": "4.9",
        "matchLat": "52.3701",
        "matchLng": "4.9001",
        "shotDate": "2023-01-01 00:00:00.000",
        "dateAdded": "2023-01-02 10:11:12",
        "heading": "74.00",
        "fieldOfView": "0",
        "gpsAccuracy": "9.3000",
        "cameraParameters": None,
        "projection": "PLANE",
        "qualityLevel": "2",
        "sequenceId": "3879193",
        "sequenceIndex": "2982",
        "wayId": "37227307",
        "width": "3840",
        "height": "2160",
        "fileurlProc": "https://example.com/proc/image1.jpg",
        "fileurlLTh": "https://example.com/lth/image1.jpg",
        "fileurlTh": "https://example.com/th/image1.jpg",
        "username": "someone",
    }


def test_fetch_metadata_bbox(fake_kartaview_client):
    df = fake_kartaview_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert not df.empty
    assert "geometry" in df.columns
    assert df.iloc[0]["id"] == 1234567890


def test_fetch_metadata_bbox_matches_db_schema(fake_kartaview_client):
    """The DataFrame columns should mirror the `kartaview` table exactly."""
    df = fake_kartaview_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert list(df.columns) == list(Project.core_tables["kartaview"]["schema"])


def test_fetch_metadata_bbox_empty(fake_kartaview_client, monkeypatch):
    """An empty bounding box still yields a DataFrame with the schema's columns."""
    monkeypatch.setattr(
        type(fake_kartaview_client), "_list_bbox", lambda self, bbox, *a, **kw: []
    )
    df = fake_kartaview_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert df.empty
    assert list(df.columns) == list(Project.core_tables["kartaview"]["schema"])


@pytest.fixture
def paged_listing(monkeypatch):
    """Serve a listing of full pages alternating flat photos and panoramas."""
    pages = []

    def fake_request(self, method, url, **kwargs):
        page = int(kwargs["files"]["page"][1])
        size = int(kwargs["files"]["ipp"][1])
        pages.append(page)
        # Page 3 is the last, and short.
        count = size if page < 3 else size // 2
        items = [
            {
                "id": str(page * 10_000 + i),
                "projection": "SPHERE" if i % 2 else "PLANE",
            }
            for i in range(count)
        ]
        return {"currentPageItems": items}

    monkeypatch.setattr(KartaViewClient, "_request", fake_request)
    return pages


def test_list_pano_only_keeps_panoramas(paged_listing):
    photos = KartaViewClient()._list_bbox((4.89, 52.37, 4.91, 52.38), 0, True)

    assert photos
    assert {photo["projection"] for photo in photos} == {"SPHERE"}
    # A filtered page is not a short page: paging runs to the real last one.
    assert paged_listing == [1, 2, 3]


def test_list_pano_only_limit_counts_panoramas(paged_listing):
    """Paging continues until the limit is met by panoramas, not by photos."""
    limit = KartaViewClient.PAGE_SIZE
    photos = KartaViewClient()._list_bbox((4.89, 52.37, 4.91, 52.38), limit, True)

    assert len(photos) == limit
    assert {photo["projection"] for photo in photos} == {"SPHERE"}
    assert paged_listing == [1, 2]


def test_list_pano_only_requests_full_pages(paged_listing):
    """A small limit must not shrink the pages that are being filtered."""
    photos = KartaViewClient()._list_bbox((4.89, 52.37, 4.91, 52.38), 5, True)

    assert len(photos) == 5
    # The first full page already holds enough panoramas.
    assert paged_listing == [1]


def test_fetch_metadata_bbox_adds_uploader(fake_kartaview_client):
    """The uploader is only listed, so it has to be carried into the record."""
    df = fake_kartaview_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert df.iloc[0]["creator"] == '{"username":"someone"}'


def test_unreachable_api_raises(monkeypatch):
    """An unreachable API must not look like a bounding box without images.

    The KartaView API is intermittently unavailable, and reporting zero images
    for a transient failure would be indistinguishable from no coverage.
    """
    attempts = []

    def timeout(self, method, url, **kwargs):
        attempts.append(url)
        raise requests.ReadTimeout("timed out")

    monkeypatch.setattr(requests.Session, "request", timeout)
    monkeypatch.setattr("streetscapes.sources.kartaview.sleep", lambda _: None)

    client = KartaViewClient(retries=3)
    with pytest.raises(KartaViewError, match="did not respond after 3 attempts"):
        client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert len(attempts) == 3


def test_retries_recover_from_a_failure(monkeypatch):
    """A single failed attempt is retried rather than propagated."""
    calls = []

    def flaky(self, method, url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ReadTimeout("timed out")
        return _ok_response()

    monkeypatch.setattr(requests.Session, "request", flaky)
    monkeypatch.setattr("streetscapes.sources.kartaview.sleep", lambda _: None)

    client = KartaViewClient(retries=3)
    assert client._request("POST", client.LIST_URL) == {"currentPageItems": []}
    assert len(calls) == 2


def _ok_response():
    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"currentPageItems": []}

    return _Response()


def test_model_matches_db_schema():
    """The model and the `kartaview` table must not drift apart."""
    schema = Project.core_tables["kartaview"]["schema"]

    # `image` is not part of the API response and KartaView reports no altitude,
    # so neither is a model field.
    assert set(KartaViewImage.model_fields) | {"image", "altitude"} == set(schema)


def test_model_conversions(valid_record):
    image = KartaViewImage.model_validate(valid_record)

    assert image.id == 1234567890
    assert image.geometry == "POINT (4.9 52.37)"
    assert image.computed_geometry == "POINT (4.9001 52.3701)"
    # KartaView reports its timestamps as UTC without saying so.
    assert image.captured_at.isoformat() == "2023-01-01T00:00:00+00:00"
    assert image.uploaded_at.isoformat() == "2023-01-02T10:11:12+00:00"
    assert image.compass_angle == 74.0
    assert image.sequence == "3879193"
    assert image.sequence_index == 2982
    # The `creator` column is JSON, so nested objects are serialised.
    assert image.creator == '{"username":"someone"}'


def test_model_derives_pano_from_projection(valid_record):
    """KartaView has no `is_pano` flag; it follows from the projection."""
    plane = KartaViewImage.model_validate(valid_record)
    sphere = KartaViewImage.model_validate({**valid_record, "projection": "SPHERE"})

    # The projection itself is stored as KartaView reports it.
    assert (plane.projection, plane.is_pano) == ("PLANE", False)
    assert (sphere.projection, sphere.is_pano) == ("SPHERE", True)


def test_model_drops_unmatched_position(valid_record):
    """An unmatched photo is reported at (0, 0) rather than as a NULL."""
    unmatched = {"matchLat": "0.000000000000000", "matchLng": "0.000000000000000"}
    image = KartaViewImage.model_validate({**valid_record, **unmatched})

    assert image.computed_geometry is None


def test_model_drops_placeholder_timestamp(valid_record):
    """A zeroed timestamp is a missing one, not a reason to drop the image."""
    image = KartaViewImage.model_validate(
        {**valid_record, "shotDate": "0000-00-00 00:00:00"}
    )

    assert image.captured_at is None
    assert image.id == 1234567890


def test_model_defaults_missing_fields():
    """Fields the API leaves out become NULLs rather than missing columns."""
    image = KartaViewImage.model_validate({"id": 1, "lat": 52.37, "lng": 4.9})

    assert image.image_url is None
    assert image.computed_geometry is None
    assert image.is_pano is None
    assert set(image.to_row()) == set(Project.core_tables["kartaview"]["schema"])
    # KartaView reports no altitude, but the column is part of the schema.
    assert image.to_row()["altitude"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "https://example.com/image1.jpg"),
        ("id", -1),
        ("lat", "952.37"),
        ("lat", "north"),
        ("lat", None),
        ("fileurlProc", "12345"),
        ("width", "wide"),
        ("shotDate", "yesterday"),
        ("sequenceIndex", -1),
    ],
)
def test_model_rejects_bad_fields(valid_record, field, value):
    with pytest.raises(ValidationError):
        KartaViewImage.model_validate({**valid_record, field: value})


def test_model_requires_id_and_position(valid_record):
    for field in ("id", "lat", "lng"):
        record = {k: v for k, v in valid_record.items() if k != field}
        with pytest.raises(ValidationError):
            KartaViewImage.model_validate(record)


def test_validate_records_skips_bad_images(valid_record):
    """A malformed image is dropped without discarding the rest of the batch."""
    records = [
        valid_record,
        {**valid_record, "id": "https://example.com/image2.jpg"},
        {**valid_record, "id": "2", "lat": "952.37"},
        {**valid_record, "id": "3"},
    ]

    images = validate_records(records)

    assert [image.id for image in images] == [1234567890, 3]
