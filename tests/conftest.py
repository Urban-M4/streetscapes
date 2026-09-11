import pandas as pd
import pytest
from pathlib import Path
from streetscapes import config
from streetscapes.sources.kartaview import KartaViewClient
from streetscapes.sources.mapillary import MapillaryClient
from streetscapes.sources.panoramax import PanoramaxClient


@pytest.fixture(autouse=True)
def test_config(tmp_path, monkeypatch):
    """Patch conf.project_dir to point to a temporary path for all tests."""
    config.CFG.project_dir = Path(tmp_path)
    config.CFG.active_project = "test_streetscapes"


@pytest.fixture
def fake_mapillary_data():
    return pd.DataFrame(
        [
            {
                "id": "1",
                "geometry": "POINT(4.89 52.37)",
                "computed_geometry": "POINT(4.89 52.37)",
            },
            {
                "id": "2",
                "geometry": "POINT(4.91 52.39)",
                "computed_geometry": "POINT(4.91 52.39)",
            },
        ]
    )

@pytest.fixture
def fake_mapillary_client(monkeypatch):
    """Patch only the API-call methods of MapillaryClient, keep the rest intact."""

    # Fake implementations
    def fake_fetch_bbox(self, bbox, limit=1000):
        return [
            {
                "id": "1",
                "geometry": {"coordinates": [4.9, 52.37]},
                "computed_geometry": {"coordinates": [4.9, 52.37]},
                "captured_at": 1672531200000,  # ms since epoch
                "thumb_2048_url": "https://example.com/image1.jpg",
            }
        ]

    def fake_download_image(self, url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FAKE IMAGE DATA")  # simulate an image file

    # Patch the class methods
    monkeypatch.setattr(MapillaryClient, "_fetch_bbox", fake_fetch_bbox)
    monkeypatch.setattr(MapillaryClient, "download_image", fake_download_image)

    # Return a real instance for convenience
    return MapillaryClient(token="fake_token")


@pytest.fixture
def fake_kartaview_client(monkeypatch):
    """Patch only the API-call methods of KartaViewClient, keep the rest intact.

    The two endpoints are patched separately so that the merge of the listing
    into the detailed records is still exercised.
    """

    def fake_list_bbox(self, bbox, limit=1000):
        # The listing endpoint reports the uploader, the photo endpoint doesn't.
        return [{"id": "1234567890", "username": "someone"}]

    def fake_fetch_details(self, image_ids):
        return [
            {
                "id": image_id,
                "lat": "52.37",
                "lng": "4.9",
                "matchLat": "52.3701",
                "matchLng": "4.9001",
                "shotDate": "2023-01-01 00:00:00.000",
                "dateAdded": "2023-01-02 10:11:12",
                "heading": "74.00",
                "projection": "PLANE",
                "width": "3840",
                "height": "2160",
                "fileurlProc": "https://example.com/proc/image1.jpg",
                "fileurlLTh": "https://example.com/lth/image1.jpg",
                "fileurlTh": "https://example.com/th/image1.jpg",
            }
            for image_id in image_ids
        ]

    def fake_download_image(self, url, output_dir, image_id, uid=None, **kwargs):
        path = Path(output_dir) / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FAKE IMAGE DATA")  # simulate an image file

    monkeypatch.setattr(KartaViewClient, "_list_bbox", fake_list_bbox)
    monkeypatch.setattr(KartaViewClient, "_fetch_details", fake_fetch_details)
    monkeypatch.setattr(KartaViewClient, "download_image", fake_download_image)

    return KartaViewClient()


@pytest.fixture
def panoramax_item():
    """A STAC item as returned by the Panoramax search endpoint."""
    return {
        "id": "2ad2b303-59f5-4f8b-a7c1-ccfaef465fbe",
        "collection": "9802529e-5ec8-433a-95ac-e9d37bf3be7e",
        "geometry": {"type": "Point", "coordinates": [2.3425668, 48.8582587]},
        "properties": {
            "datetime": "2024-10-05T14:24:57+00:00",
            "created": "2024-10-05T18:24:28.811997+00:00",
            "view:azimuth": 280,
            "license": "etalab-2.0",
            "geovisio:producer": "tdelmas",
            "geovisio:rank_in_collection": 344,
            "quality:horizontal_accuracy": 4.0,
            "pers:interior_orientation": {
                "field_of_view": 360,
                "sensor_array_dimensions": [5760, 2880],
            },
        },
        "assets": {
            "hd": {"href": "https://example.com/pictures/hd.jpg"},
            "sd": {"href": "https://example.com/pictures/sd.jpg"},
            "thumb": {"href": "https://example.com/pictures/thumb.jpg"},
        },
        "links": [
            {"rel": "self", "href": "https://api.panoramax.xyz/api/items/x"},
            {"rel": "via", "href": "https://panoramax.ign.fr"},
        ],
    }


@pytest.fixture
def fake_panoramax_client(monkeypatch, panoramax_item):
    """Patch only the API call of PanoramaxClient, keep the rest intact."""

    def fake_request(self, params):
        return {"features": [panoramax_item]}

    def fake_download_image(self, url, output_dir, image_id, uid=None, **kwargs):
        path = Path(output_dir) / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("FAKE IMAGE DATA")  # simulate an image file

    monkeypatch.setattr(PanoramaxClient, "_request", fake_request)
    monkeypatch.setattr(PanoramaxClient, "download_image", fake_download_image)

    return PanoramaxClient()
