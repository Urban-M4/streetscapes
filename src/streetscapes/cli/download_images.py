import os
import typer
from pathlib import Path
import pandas as pd
from streetscapes.sources.downloader import ImageDownloader
from streetscapes.sources.mapillary import Mapillary
from streetscapes.sources.amsterdam import AmsterdamPanorama
from streetscapes.workspace import Workspace


download_images_cli = typer.Typer(help="Download images from a source using a manifest")


def _load_manifest(manifest_path: Path):
    return pd.read_parquet(manifest_path)


@download_images_cli.command("mapillary")
def download_mapillary(
    manifest_path: Path = typer.Argument(
        None, help="Manifest file ((geo)parquet from fetch_metadata)"
    ),
    output_dir: Path = typer.Option(
        None,
        help="Base output directory (default: STREETSCAPES_OUTPUT_DIR from env or ./streetscapes_output)",
    ),
    overwrite: bool = typer.Option(False, help="Overwrite existing images"),
    limit: int = typer.Option(None, help="Limit number of images to download"),
    token: str = typer.Option(
        None, help="Mapillary OAuth token (or set MAPILLARY_TOKEN env var)"
    ),
):
    ws = Workspace.from_env() if output_dir is None else Workspace(Path(output_dir))
    token = token or os.getenv("MAPILLARY_TOKEN")
    if not token:
        typer.echo(
            "Error: Mapillary token not provided and MAPILLARY_TOKEN not set in .env.",
            err=True,
        )
        raise typer.Exit(code=2)
    images_dir = ws.images
    manifest_dir = ws.manifests

    # Use default manifest path if not provided
    if manifest_path is None:
        manifest_path = manifest_dir / "fetch_metadata_mapillary.parquet"

    df = _load_manifest(manifest_path)
    if limit is not None:
        df = df.head(limit)
    source = Mapillary(token)
    downloader = ImageDownloader(
        source, manifest_dir=manifest_dir, images_dir=images_dir
    )
    downloader.download_from_manifest(
        df, id_column="id", url_column="thumb_2048_url", overwrite=overwrite
    )
    typer.echo(f"Downloaded {len(df)} Mapillary images to {images_dir}")


@download_images_cli.command("amsterdam")
def download_amsterdam(
    manifest_path: Path = typer.Argument(..., help="Manifest file (CSV or Parquet)"),
    output_dir: Path = typer.Option("images", help="Directory to store images"),
    overwrite: bool = typer.Option(False, help="Overwrite existing images"),
):
    df = _load_manifest(manifest_path)
    image_ids = df["pano_id"].astype(str).tolist()
    source = AmsterdamPanorama()
    downloader = ImageDownloader(source, output_dir=output_dir)
    downloader.download(image_ids, overwrite=overwrite)
    typer.echo(f"Downloaded {len(image_ids)} Amsterdam Panorama images to {output_dir}")
