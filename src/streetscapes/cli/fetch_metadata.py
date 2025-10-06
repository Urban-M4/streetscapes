import typer
import logging

logger = logging.getLogger(__name__)

fetch_metadata_cli = typer.Typer(help="Fetch metadata for a source")

def _get_mapillary_client(token):
    """Handle lazy import of MapillaryClient."""
    from streetscapes.sources.mapillary import MapillaryClient

    return MapillaryClient(token)


Bbox = tuple[float, float, float, float]
"""west, south, easth, north."""


@fetch_metadata_cli.command("mapillary")
def fetch_metadata_mapillary(
    bbox: Bbox = typer.Option(..., help="Bounding box (west, south, east, north)"),  # noqa: B008
    tile_size: float = typer.Option(0.001, help="Tile size in degrees"),
    limit: int = typer.Option(1000, help="Maximum number of images per tile"),
    token: str = typer.Option(None, help="Mapillary OAuth token."),
    project_path: str = typer.Option(
        "streetscapes.duckdb", "--project", help="Name of the current project."
    ),
):
    """Fetch Mapillary metadata in tiles and store as DuckDB manifest."""
    import os

    import ibis

    from rich.progress import track
    from streetscapes.cli.console import console
    from streetscapes.project import Project
    from streetscapes.utils.bbox import split_bbox

    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        logger.error("Error: token not provided and MAPILLARY_TOKEN not set in .env.")
        raise typer.Exit(code=1)

    logger.info(f"Fetching metadata for {bbox=}")
    m = _get_mapillary_client(token)

    project = Project(project_path)

    ntiles, tiles = split_bbox(bbox, tile_size)
    logger.info(f"Splitting bbox in {ntiles} tiles with {tile_size=}")
    for tile, tile_id in track(
        tiles, description="Fetching tiles", total=ntiles, console=console
    ):
        df = m.fetch_metadata_bbox(tile, limit)

        # TODO: maybe this failsafe/optimization is not necessary?
        if len(df) == 0:
            continue

        project.ingest_mapillary(df)

    # Inform user about result
    ibis.options.interactive = True
    filtered = project.filter_bbox("mapillary", bbox)
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