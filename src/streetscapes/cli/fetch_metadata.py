import os

from rich import print
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
    from streetscapes.sources.mapillary import MapillaryClient

    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        print("Error: token not provided and MAPILLARY_TOKEN not set in .env.")
        raise typer.Exit(code=1)

    print(f"Fetching Mapillary metadata for bbox={bbox}")

    m = MapillaryClient(token)
    df = m.fetch_metadata(bbox=bbox, tile_size=tile_size, limit=limit)

    print(df.head())
