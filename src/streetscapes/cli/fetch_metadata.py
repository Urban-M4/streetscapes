"""Metadata fetching CLI.

Usage:
> streetscapes fetch-metadata --help
"""

import logging
from typing import TYPE_CHECKING

from cyclopts import App
from rich.progress import track

from streetscapes import CFG
from streetscapes.cli.console import console
from streetscapes.utils.geo import Bbox, split_bbox

if TYPE_CHECKING:
    from streetscapes.project import Project

logger = logging.getLogger(__name__)

fetch_metadata_cli = App(help="Fetch metadata for a source")


@fetch_metadata_cli.command(name="mapillary")
def mapillary(
    bbox: Bbox,
    /,
    *,
    tile_size: float = 0.001,
    tile_limit: int = 1000,
    token: str | None = None,
    project: str | None = None,
):
    """Fetch metadata from the Mapillary API.

    Args:
        bbox: Bounding box (WEST SOUTH EAST NORTH).
        tile_size: Tile size in degrees.
        tile_limit: Maximum number of images per tile.
        token: Mapillary OAuth token (if not set via MAPILLARY_TOKEN).
        project: An optional project to attach to.
    """
    from streetscapes.project import Project
    from streetscapes.sources.mapillary import MapillaryClient

    logger.info(f"Fetching metadata for {bbox=}")

    token = token or CFG.mapillary_token
    if not token:
        logger.error(
            "Error: 'mapillary_token' missing, set with `streetscapes config set"
            " mapillary_token <your token>`"
        )
        raise SystemExit(1)

    m = MapillaryClient(token)
    proj = Project(project)

    ntiles, tiles = split_bbox(bbox, tile_size)
    logger.info(f"Splitting bbox in {ntiles} tiles with {tile_size=}")
    for tile, _tile_id in track(
        tiles, description="Fetching tiles", total=ntiles, console=console
    ):
        df = m.fetch_metadata_bbox(tile, tile_limit)

        # TODO: maybe this failsafe/optimization is not necessary?
        if len(df) == 0:
            continue

        proj.ingest_metadata(df, "mapillary")

    _report(proj, "mapillary", bbox)


@fetch_metadata_cli.command(name="kartaview")
def kartaview(
    bbox: Bbox,
    /,
    *,
    image_limit: int = 1000,
    project: str | None = None,
):
    """Fetch metadata from the KartaView API.

    Args:
        bbox: Bounding box (WEST SOUTH EAST NORTH).
        image_limit: Maximum number of images to fetch (0 for no limit).
        project: An optional project to attach to.
    """
    from streetscapes.project import Project
    from streetscapes.sources.kartaview import KartaViewClient, KartaViewError

    logger.info(f"Fetching metadata for {bbox=}")

    client = KartaViewClient()
    proj = Project(project)

    try:
        with console.status("Fetching images..."):
            df = client.fetch_metadata_bbox(bbox, image_limit)
    except KartaViewError as err:
        logger.error(str(err))
        raise SystemExit(1) from err

    logger.info(f"Fetched metadata for {len(df)} images.")
    proj.ingest_metadata(df, "kartaview")

    _report(proj, "kartaview", bbox)


def _report(proj: "Project", table: str, bbox: Bbox):
    """Show the user what ended up in the table for the requested bounding box."""
    import ibis

    ibis.options.interactive = True
    filtered = proj.filter_bbox(table, bbox)
    logger.info(f"Total images in bbox: {filtered.count().execute()}, first 5 rows:")
    console.print(filtered.limit(5))  # console print gives nicer table than logger
    logger.info("Ready.")


# To check the table:
# import ibis
# ibis.options.interactive = True
# db = ibis.duckdb.connect("streetscapes.duckdb")
# tab = db.table('mapillary_data')
# print(tab.count())
# print(tab.nunique())


# TODO: consider re-implementing crash recovery by keeping track of
# which tiles have already been ingested? Could use a temporary
# table "processed_tiles", skip tiles from that table, and drop it
# when the CLI completes successfully.
