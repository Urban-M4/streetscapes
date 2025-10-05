import typer
from streetscapes.utils.bbox import Bbox, split_bbox


fetch_metadata_cli = typer.Typer(help="Fetch metadata for a source")


@fetch_metadata_cli.command("mapillary")
def fetch_metadata_mapillary(
    bbox: Bbox = typer.Option(..., help="Bounding box (west, south, east, north)"),  # noqa: B008
    tile_size: float = typer.Option(0.001, help="Tile size in degrees"),
    limit: int = typer.Option(1000, help="Maximum number of images per tile"),
    token: str = typer.Option(None, help="Mapillary OAuth token."),
):
    """Fetch Mapillary metadata in tiles and store as DuckDB manifest."""
    import os

    import ibis
    from rich import print

    from rich.progress import track
    from streetscapes.cli.console import console
    from streetscapes.sources.mapillary import MapillaryClient

    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        print("Error: token not provided and MAPILLARY_TOKEN not set in .env.")
        raise typer.Exit(code=1)

    print(f"Fetching Mapillary metadata for bbox={bbox}")
    m = MapillaryClient(token)

    db = ibis.duckdb.connect("streetscapes.duckdb")
    db.raw_sql("INSTALL spatial; LOAD spatial;")

    ntiles, tiles = split_bbox(bbox, tile_size)
    for tile, tile_id in track(
        tiles, description="Fetching Mapillary tiles", total=ntiles, console=console
    ):
        df = m.fetch_metadata_bbox(tile, limit)

        # TODO: maybe this failsafe/optimization is not necessary?
        if len(df) == 0:
            continue

        # TODO: consider re-implementing crash recovery by keeping track of
        # which tiles have already been ingested? Could use a temporary
        # table "processed_tiles", skip tiles from that table, and drop it
        # when the CLI completes successfully.

        db.con.register("metadata_tile", df)

        if "mapillary_data" not in db.list_tables():
            db.raw_sql("""
                CREATE TABLE mapillary_data AS
                SELECT
                    * EXCLUDE (geometry, computed_geometry),
                    ST_GeomFromText(geometry) AS geometry,
                    ST_GeomFromText(computed_geometry) AS computed_geometry
                FROM metadata_tile;
                    
                ALTER TABLE mapillary_data
                ADD PRIMARY KEY (id);
            """)
        else:
            # TODO: consider adding duplicate behaviour (REPLACE or IGNORE) as
            # CLI option
            db.raw_sql("""
                INSERT OR REPLACE INTO mapillary_data
                SELECT
                    * EXCLUDE (geometry, computed_geometry),
                    ST_GeomFromText(geometry) AS geometry,
                    ST_GeomFromText(computed_geometry) AS computed_geometry
                FROM metadata_tile
            """)

    # preview metadata table
    ibis.options.interactive = True
    print(db.table("mapillary_data"))


# To check the table:
# import ibis
# ibis.options.interactive = True
# db = ibis.duckdb.connect("streetscapes.duckdb")
# tab = db.table('mapillary_data')
# print(tab.count())
# print(tab.nunique())