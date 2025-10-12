import pandas as pd
import pytest

from streetscapes import config
from streetscapes.sources.mapillary import MapillaryClient


@pytest.fixture(autouse=True)
def patch_data_home(tmp_path, monkeypatch):
    """Patch config.get('data_home') to point to a temporary path for all tests."""
    patched_config = {
        "data_home": str(tmp_path),
        "active_project": "test_streetscapes",
    }
    monkeypatch.setattr(
        config,
        "get",
        patched_config.get,
    )


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
                "captured_at": "2023-01-01T00:00:00Z",
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