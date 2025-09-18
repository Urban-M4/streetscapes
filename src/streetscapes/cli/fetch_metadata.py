import typer
from pathlib import Path
from streetscapes.sources.mapillary import Mapillary
from streetscapes.sources.amsterdam import AmsterdamPanorama
from streetscapes.sources.mapillary import PyArrowGeoParquetWriter
import os
from dotenv import load_dotenv

fetch_metadata_cli = typer.Typer(help="Fetch metadata for a source")


@fetch_metadata_cli.command("mapillary")
def fetch_metadata_mapillary(
    bbox: list[float] = typer.Option(
        ..., help="Bounding box [west, south, east, north]"
    ),
    tile_size: float = typer.Option(0.01, help="Tile size in degrees"),
    output_file: Path = typer.Option(..., help="Output GeoParquet file"),
    token: str = typer.Option(
        None, help="Mapillary OAuth token (optional, will use .env if not provided)"
    ),
):
    """Fetch Mapillary metadata in tiles and store as GeoParquet."""
    load_dotenv()
    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        typer.echo(
            "Error: Mapillary token not provided and MAPILLARY_TOKEN not set in .env.",
            err=True,
        )
        raise typer.Exit(code=1)
    source = Mapillary(token)
    writer = PyArrowGeoParquetWriter()
    table = source.fetch_metadata(bbox, tile_size, output_file, writer=writer)
    typer.echo(f"Saved {len(table)} records to {output_file}")


@fetch_metadata_cli.command("amsterdam")
def fetch_metadata_amsterdam(
    lat: float = typer.Option(..., help="Latitude"),
    lon: float = typer.Option(..., help="Longitude"),
    radius: float = typer.Option(50.0, help="Radius in meters"),
    output_file: Path = typer.Option(..., help="Output GeoParquet file"),
):
    """Fetch Amsterdam Panorama metadata and store as GeoParquet."""
    source = AmsterdamPanorama()
    table = source.fetch_metadata(lat, lon, radius, output_file)
    typer.echo(f"Saved {len(table)} records to {output_file}")
