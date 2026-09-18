"""Tests for the image download loop shared by the source clients."""

import threading
import uuid

import pandas as pd
import pytest

from streetscapes.cli.download_images import _download_images
from streetscapes.project import Project
from streetscapes.sources.mapillary import MapillaryImage
from streetscapes.utils.metadata import ImageMeta

# A one-pixel PNG, so that what is written to disk is a real image.
PIXEL = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c63000100000500010d0a2db40000"
    "000049454e44ae426082"
)


class FakeClient:
    """A source client that writes an image without going near the network."""

    def __init__(self, failing: set | None = None):
        self.failing = failing or set()
        self.threads: set[int] = set()
        self.lock = threading.Lock()

    def download_image(self, url, output_dir, image_id, uid=None, skip_existing=True):
        with self.lock:
            self.threads.add(threading.get_ident())

        if image_id in self.failing:
            raise RuntimeError(f"failed to download {image_id}")

        # A UUID of the image's own, as a real download would derive from content.
        image_uid = uuid.UUID(int=int(image_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{image_uid}.png").write_bytes(PIXEL)
        (output_dir / str(image_id)).write_text(str(image_uid))

        return ImageMeta(PIXEL, b"", image_uid, "png", fpath=output_dir)


@pytest.fixture
def project(tmp_path):
    """A project holding a handful of Mapillary records, none downloaded yet."""
    proj = Project(
        "test_downloads", image_dir=tmp_path / "images", project_dir=tmp_path
    )
    rows = [
        MapillaryImage.model_validate(
            {
                "id": image_id,
                "geometry": {"coordinates": [4.9 + image_id / 1000, 52.37]},
                "thumb_2048_url": f"https://example.com/{image_id}.jpg",
                "is_pano": image_id % 2 == 0,
            }
        ).to_row()
        for image_id in range(1, 11)
    ]
    proj.ingest_metadata(
        pd.DataFrame(rows, columns=list(proj.schema("mapillary"))), "mapillary"
    )
    return proj


def _records(proj):
    return proj.get_download_records("mapillary", "thumb_2048_url", True)


@pytest.mark.parametrize("workers", [1, 8])
def test_downloads_and_registers_every_image(project, workers):
    client = FakeClient()
    records = _records(project)
    assert len(records) == 10

    _download_images(project, client, "mapillary", records, True, workers)

    # Every image is on disk, registered, and linked from its source row.
    assert len(list((project.image_dir / "images").rglob("*.png"))) == 10
    assert project.table("images").count().execute() == 10
    assert not _records(project)


def test_downloads_run_in_several_threads(project):
    client = FakeClient()

    _download_images(project, client, "mapillary", _records(project), True, workers=8)

    assert len(client.threads) > 1


def test_downloads_stay_on_the_calling_thread_without_workers(project):
    client = FakeClient()

    _download_images(project, client, "mapillary", _records(project), True, workers=1)

    assert client.threads == {threading.get_ident()}


@pytest.mark.parametrize("workers", [1, 8])
def test_a_failed_download_does_not_stop_the_others(project, workers):
    client = FakeClient(failing={3, 7})
    records = _records(project)

    _download_images(project, client, "mapillary", records, True, workers)

    assert project.table("images").count().execute() == 8
    # The two that failed are still waiting to be downloaded.
    assert sorted(rec[1] for rec in _records(project)) == [3, 7]


def test_tags_mark_the_panoramas(project):
    """Which images are panoramic has to survive being downloaded out of order."""
    _download_images(project, FakeClient(), "mapillary", _records(project), True, 8)

    images = project.table("images").to_pandas()
    panoramic = {
        str(row.uuid) for row in images.itertuples() if "panoramic" in row.tags
    }

    assert panoramic == {str(uuid.UUID(int=i)) for i in (2, 4, 6, 8, 10)}
