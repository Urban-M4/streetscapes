import pytest
import requests
from pydantic import ValidationError

from streetscapes.project import Project
from streetscapes.sources.kartaview import (
    KartaViewClient,
    KartaViewError,
    KartaViewImage,
    filter_listed,
    validate_records,
)


@pytest.fixture
def valid_record():
    """An image record as returned by the KartaView photo endpoint.

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
    """Serve a listing of full pages alternating flat images and panoramas.

    Every other pair of images is captured at night.
    """
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
                "lat": "52.37",
                "lng": "4.9",
                "shot_date": f"2025-01-15 {'12' if i % 4 < 2 else '20'}:00:00.000",
                "projection": "SPHERE" if i % 2 else "PLANE",
            }
            for i in range(count)
        ]
        return {"currentPageItems": items}

    monkeypatch.setattr(KartaViewClient, "_request", fake_request)
    return pages


def test_list_pano_only_keeps_panoramas(paged_listing):
    images = KartaViewClient()._list_bbox((4.89, 52.37, 4.91, 52.38), 0, True)

    assert images
    assert {image["projection"] for image in images} == {"SPHERE"}
    # A filtered page is not a short page: paging runs to the real last one.
    assert paged_listing == [1, 2, 3]


def test_list_pano_only_limit_counts_panoramas(paged_listing):
    """Paging continues until the limit is met by panoramas, not by images."""
    limit = KartaViewClient.PAGE_SIZE
    images = KartaViewClient()._list_bbox((4.89, 52.37, 4.91, 52.38), limit, True)

    assert len(images) == limit
    assert {image["projection"] for image in images} == {"SPHERE"}
    assert paged_listing == [1, 2]


def test_list_pano_only_requests_full_pages(paged_listing):
    """A small limit must not shrink the pages that are being filtered."""
    images = KartaViewClient()._list_bbox((4.89, 52.37, 4.91, 52.38), 5, True)

    assert len(images) == 5
    # The first full page already holds enough panoramas.
    assert paged_listing == [1]


def test_list_daytime_only_limit_counts_daytime(paged_listing):
    """Paging continues until the limit is met by daytime images."""
    limit = KartaViewClient.PAGE_SIZE
    images = KartaViewClient()._list_bbox(
        (4.89, 52.37, 4.91, 52.38), limit, daytime_only=True
    )

    assert len(images) == limit
    assert {image["shot_date"][11:13] for image in images} == {"12"}
    assert paged_listing == [1, 2]


def test_list_combines_filters(paged_listing):
    images = KartaViewClient()._list_bbox(
        (4.89, 52.37, 4.91, 52.38), 5, pano_only=True, daytime_only=True
    )

    assert len(images) == 5
    assert {(p["projection"], p["shot_date"][11:13]) for p in images} == {
        ("SPHERE", "12")
    }
    assert paged_listing == [1]


def test_filter_listed_daytime_drops_what_cannot_be_placed():
    """A listed image without a usable capture time or position is dropped."""
    day = {"id": "1", "lat": "52.37", "lng": "4.9", "shot_date": "2025-01-15 12:00:00"}
    images = [
        day,
        {**day, "id": "2", "shot_date": "2025-01-15 17:30:00"},
        {**day, "id": "3", "shot_date": "0000-00-00 00:00:00"},
        {**day, "id": "4", "shot_date": None},
        {**day, "id": "5", "lat": "0", "lng": "0"},
    ]

    assert filter_listed(images, daytime_only=True) == [day]
    assert filter_listed(images) == images


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


def test_token_is_sent_as_query_parameter(monkeypatch):
    """An access token raises the rate limit, and goes in the query string."""
    calls = []

    def record(self, method, url, **kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr(requests.Session, "request", record)

    KartaViewClient("secret")._request(
        "GET", KartaViewClient.DETAIL_URL, params={"id": "1"}
    )
    KartaViewClient()._request("GET", KartaViewClient.DETAIL_URL, params={"id": "1"})

    assert calls[0]["params"] == {"id": "1", "access_token": "secret"}
    assert calls[1]["params"] == {"id": "1"}


def test_token_is_not_logged(monkeypatch, caplog):
    """Errors quote the request URL, which must not leak the token."""

    def fail(self, method, url, **kwargs):
        raise requests.HTTPError(
            f"429 Too Many Requests for url: {url}?access_token=secret"
        )

    monkeypatch.setattr(requests.Session, "request", fail)
    monkeypatch.setattr("streetscapes.sources.kartaview.sleep", lambda _: None)

    with pytest.raises(KartaViewError):
        KartaViewClient("secret", retries=1)._request("GET", KartaViewClient.DETAIL_URL)

    assert "429" in caplog.text
    assert "secret" not in caplog.text


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
    """An unmatched image is reported at (0, 0) rather than as a NULL."""
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
