fetch_metadata_cli = None


import typer

fetch_metadata_cli = typer.Typer(help="Fetch metadata for a source")


@fetch_metadata_cli.command("mapillary")
def fetch_metadata_mapillary(
    bbox: tuple[float, float, float, float] = typer.Option(
        ..., help="Bounding box [west, south, east, north]"
    ),
    tile_size: float = typer.Option(0.01, help="Tile size in degrees"),
    output_dir=typer.Option(
        None,
        help="Base output directory (default: STREETSCAPES_OUTPUT_DIR or ./streetscapes_output)",
    ),
    token: str = typer.Option(
        None, help="Mapillary OAuth token (optional, will use .env if not provided)"
    ),
):
    import logging
    import os
    from pathlib import Path
    from streetscapes.workspace import Workspace
    from streetscapes.sources.mapillary import Mapillary

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ws = Workspace.from_env() if output_dir is None else Workspace(Path(output_dir))
    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        typer.echo(
            "Error: Mapillary token not provided and MAPILLARY_TOKEN not set in .env.",
            err=True,
        )
        raise typer.Exit(code=1)
    manifest_db_path = ws.manifests / "fetch_metadata_mapillary.duckdb"
    logging.info(f"Using token: {'***' + token[-6:] if token else None}")
    logging.info(
        f"Starting Mapillary metadata fetch for bbox={bbox}, tile_size={tile_size}, manifest_db={manifest_db_path}"
    )
    source = Mapillary(token)
    table = source.fetch_metadata(bbox, tile_size, manifest_db_path)
    if table is not None and len(table) > 0:
        logging.info(f"Saved {len(table)} records to {manifest_db_path}")
    else:
        logging.warning(f"No records found for bbox={bbox}. Manifest may be empty.")
    typer.echo("To preview your manifest, run:")
    typer.echo(f"streetscapes manifest head {manifest_db_path}")


@fetch_metadata_cli.command("amsterdam")
def fetch_metadata_amsterdam(
    lat: float = typer.Option(..., help="Latitude"),
    lon: float = typer.Option(..., help="Longitude"),
    radius: float = typer.Option(50.0, help="Radius in meters"),
    output_file=typer.Option(..., help="Output GeoParquet file"),
):
    from streetscapes.sources.amsterdam import AmsterdamPanorama
    source = AmsterdamPanorama()
    table = source.fetch_metadata(lat, lon, radius, output_file)
    typer.echo(f"Saved {len(table)} records to {output_file}")
