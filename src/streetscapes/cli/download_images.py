import logging
import os
from pathlib import Path

import pygeohash
import typer
from rich.progress import track
from shapely import from_wkb

logger = logging.getLogger(__name__)


download_images_cli = typer.Typer(name="download_images")


def _get_mapillary_client(token=None):
    """Handle lazy import of MapillaryClient."""
    from streetscapes.sources.mapillary import MapillaryClient

    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        logger.error("Error: token not provided and MAPILLARY_TOKEN not set in .env.")
        raise typer.Exit(code=1)

    return MapillaryClient(token)


@download_images_cli.command("mapillary")
def download_mapillary(
    skip_existing: bool = typer.Option(
        True, help="If true, only download missing images, otherwise overwrite."
    ),
    token: str = typer.Option(
        None, help="Mapillary OAuth token (if not set via MAPILLARY_TOKEN)."
    ),
):
    """Download Mapillary images to a local directory."""
    from streetscapes import config
    from streetscapes.project import Project

    project = Project(config.get("active_project"))
    records = project.get_mapillary_download_records(skip_existing)

    if not records:
        typer.echo("No new images to download.")
        raise typer.Exit()

    mapillary = _get_mapillary_client(token)
    data_home = Path(config.get("data_home"))

    batch = []
    for image_id, url, geometry in track(records, "Downloading images..."):
        # Determine path
        shard = _get_geohash_shard_path(geometry)
        path = data_home / "images" / "mapillary" / shard / f"{image_id}.jpg"

        # Download image
        mapillary.download_image(url, path)

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
