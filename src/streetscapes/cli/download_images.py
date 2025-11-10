import logging
from pathlib import Path

import pygeohash
from cyclopts import App
from rich.progress import track
from shapely import from_wkb

from streetscapes import config
from streetscapes.cli.console import console
from streetscapes.project import Project
from streetscapes.sources.mapillary import MapillaryClient

logger = logging.getLogger(__name__)

download_images_cli = App(help="Download images from various sources.")


@download_images_cli.command(name="mapillary")
def mapillary(
    skip_existing: bool = True,
    token: str | None = None,
):
    """Download Mapillary images to a local directory.

    Parameters
    ----------
    skip_existing:
        If true, only download missing images; otherwise overwrite.
    token:
        Mapillary OAuth token (if not set via MAPILLARY_TOKEN).
    """
    project_name = config.get("active_project")
    data_home = Path(config.get("data_home"))

    # TODO: perhaps move this to context in main cli?
    console.rule("Streetscapes")
    console.print(f"Active project: {project_name}")
    console.print(f"Data home: {data_home}")

    project = Project(project_name)
    records = project.get_mapillary_download_records(skip_existing)

    if not records:
        print("No new images to download.")
        raise SystemExit(0)

    mapillary = MapillaryClient(token)
    base_path = data_home / "images" / "mapillary"

    total = len(records)
    console.print(f"Downloading {total} image(s) to {base_path}.")

    batch = []
    downloaded = 0
    for image_id, url, geometry in track(records, "Downloading images..."):
        # Determine path
        shard = _get_geohash_shard_path(geometry)
        path = base_path / shard / f"{image_id}.jpg"

        # Download image
        mapillary.download_image(url, path)
        downloaded += 1

        # Add metadata to batch
        batch.append(
            {"id": image_id, "source": "mapillary", "path": path, "geometry": geometry}
        )

        # Insert batch into database
        if len(batch) >= 5:
            project.ingest_local_images(batch)
            batch.clear()

    # Insert remaining records into database
    if batch:
        project.ingest_local_images(batch)

    console.print(
        f"Download complete: {downloaded}/{total} images saved under {base_path}."
    )


def _get_geohash_shard_path(geometry):
    """Get nested geo-hash path for given location.

    Geo-hash precision from
    https://python-bloggers.com/2024/02/geohashing-from-scratch-in-python/
    Precision          Dimension
            1: 5,000km x 5,000km
            2:   1,250km x 625km
            3:     156km x 156km
            4:   31.9km x 19.5km
            5:   4.89km x 4.89km
            6:   1.22km x 0.61km
            7:       153m x 153m
            8:     38.2m x 19.1m
            9:     4.77m x 4.77m
           10:    1.19m x 0.596m
           11:     149mm x 149mm
           12:   37.2mm x 18.6mm
        Each level of precision subdivides the previous level into 32 subtiles.
    Shard path of precision 7, split in three parts abc/de/fg
    abc/ --> region level
    de/ --> neighbourhood scale (max 32x32 = 1024 per region)
    fg/ --> block level  (max 32x32 = 1024 per neighbourhood)
    """
    geom = from_wkb(geometry)
    geohash = pygeohash.encode(geom.y, geom.x, precision=7)  # 153m x 153m
    return Path(geohash[:2]) / geohash[2:4] / geohash[4:6]
