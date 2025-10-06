import pytest
from rich.console import Console

from streetscapes.sources.mapillary import MapillaryClient


@pytest.fixture(autouse=True)
def no_color_console(monkeypatch):
    """Patch the CLI console to disable colors in tests."""
    plain_console = Console(force_terminal=False, color_system=None)
    monkeypatch.setattr("streetscapes.cli.console.console", plain_console)
    return plain_console


@pytest.fixture
def fake_mapillary_client(monkeypatch):
    """A mock MapillaryClient that avoids real network calls."""

    client = MapillaryClient(token="fake_token")

    def fake_fetch_bbox(bbox, limit=1000):
        return [
            {
                "id": "1",
                "geometry": {"coordinates": [4.9, 52.37]},
                "computed_geometry": {"coordinates": [4.9, 52.37]},
                "captured_at": "2023-01-01T00:00:00Z",
                "thumb_2048_url": "https://example.com/image1.jpg",
            }
        ]

    # Monkeypatch the private method just once, hidden from the test
    monkeypatch.setattr(client, "_fetch_bbox", fake_fetch_bbox)

    return client
