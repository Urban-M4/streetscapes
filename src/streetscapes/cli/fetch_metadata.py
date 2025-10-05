import typer

Bbox = tuple[float, float, float, float]
"""(west, south, east, north)"""

fetch_metadata_cli = typer.Typer(help="Fetch metadata for a source")


@fetch_metadata_cli.command("mapillary")
def fetch_metadata_mapillary(
    bbox: Bbox = typer.Option(..., help="Bounding box (west, south, east, north)"),
    tile_size: float = typer.Option(0.01, help="Tile size in degrees"),
    limit: int = typer.Option(1000, help="Maximum number of images per tile"),
    token: str = typer.Option(None, help="Mapillary OAuth token."),
):
    """Fetch Mapillary metadata in tiles and store as DuckDB manifest."""
    import os

    import ibis
    from rich import print

    from streetscapes.sources.mapillary import MapillaryClient

    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        print("Error: token not provided and MAPILLARY_TOKEN not set in .env.")
        raise typer.Exit(code=1)

    print(f"Fetching Mapillary metadata for bbox={bbox}")
    m = MapillaryClient(token)

    db = ibis.duckdb.connect("streetscapes.duckdb")
    db.raw_sql("INSTALL spatial; LOAD spatial;")

    for _, df in m.iter_metadata(bbox, tile_size, limit):
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
            db.raw_sql("""
                INSERT INTO mapillary_data
                SELECT
                    * EXCLUDE (geometry, computed_geometry),
                    ST_GeomFromText(geometry) AS geometry,
                    ST_GeomFromText(computed_geometry) AS computed_geometry
                FROM metadata_tile
            """)

    # preview metadata table
    ibis.options.interactive = True
    print(db.table("mapillary_data"))