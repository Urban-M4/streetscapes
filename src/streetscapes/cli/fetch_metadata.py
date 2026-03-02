import logging
import os

import ibis
from cyclopts import App
from rich.progress import track
import typer

from streetscapes import CFG
from streetscapes.cli.console import console
from streetscapes.project import Project
from streetscapes.sources.mapillary import MapillaryClient
from streetscapes.utils.bbox import Bbox, split_bbox

logger = logging.getLogger(__name__)

fetch_metadata_cli = App(help="Fetch metadata for a source")


@fetch_metadata_cli.command(name="mapillary")
def mapillary(
    bbox: Bbox,
    tile_size: float = 0.01,
    limit: int = 1000,
    token: str | None = None,
    project: str | None = None,
):
    """Fetch metadata from the Mapillary API.

    Args:
        bbox: Bounding box (west, south, east, north).
        tile_size: Tile size in degrees.
        limit: Maximum number of images per tile.
        token: Mapillary OAuth token (if not set via MAPILLARY_TOKEN).
        project: An optional project to attach to.
    """

    logger.info(f"Fetching metadata for {bbox=}")

    token = token or CFG.mapillary_token
    if not token:
        logger.error("Error: token not provided and MAPILLARY_TOKEN not set in .env.")
        raise typer.Exit(code=1)

    m = MapillaryClient(token)
    proj = Project(project)

    ntiles, tiles = split_bbox(bbox, tile_size)
    logger.info(f"Splitting bbox in {ntiles} tiles with {tile_size=}")
    for tile, tile_id in track(
        tiles, description="Fetching tiles", total=ntiles, console=console
    ):
        df = m.fetch_metadata_bbox(tile, limit)

        # TODO: maybe this failsafe/optimization is not necessary?
        if len(df) == 0:
            continue

        proj.ingest_mapillary(df)

    # Inform user about result
    ibis.options.interactive = True
    filtered = proj.filter_bbox("mapillary", bbox)
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
