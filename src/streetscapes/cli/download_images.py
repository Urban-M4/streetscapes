import typer
from pathlib import Path
import pandas as pd
from streetscapes.sources.downloader import ImageDownloader
from streetscapes.sources.mapillary import Mapillary
from streetscapes.sources.amsterdam import AmsterdamPanorama

download_images_cli = typer.Typer(help="Download images from a source using a manifest")


def _load_manifest(manifest_path: Path):
    if manifest_path.suffix == ".csv":
        return pd.read_csv(manifest_path)
    return pd.read_parquet(manifest_path)


@download_images_cli.command("mapillary")
def download_mapillary(
    manifest_path: Path = typer.Argument(..., help="Manifest file (CSV or Parquet)"),
    output_dir: Path = typer.Option("images", help="Directory to store images"),
    overwrite: bool = typer.Option(False, help="Overwrite existing images"),
    token: str = typer.Option(..., help="Mapillary OAuth token"),
):
    df = _load_manifest(manifest_path)
    image_ids = df["id"].astype(str).tolist()
    source = Mapillary(token)
    downloader = ImageDownloader(source, output_dir=output_dir)
    downloader.download(image_ids, overwrite=overwrite)
    typer.echo(f"Downloaded {len(image_ids)} Mapillary images to {output_dir}")


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
