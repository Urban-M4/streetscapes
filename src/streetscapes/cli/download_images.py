"""Image download CLI.

usage:
> streetscapes download-images --help
"""

import logging
from pathlib import Path

import typer
from cyclopts import App
from rich.progress import track

from streetscapes import CFG, utils
from streetscapes.cli.console import console
from streetscapes.project import _format_image

logger = logging.getLogger(__name__)

download_images_cli = App(help="Download images from various sources.")


def _validate_uuid(uid: str, output_dir: Path) -> bool:
    img_file = (output_dir / uid).with_suffix(".jpg")
    if img_file.exists() and uid == utils.get_image_uuid(img_file):
        return True
    return False


def _existing_img_valid(
    uid: str | None,
    image_id: int | None,
    output_dir: Path,
    skip_existing: bool,
) -> bool:
    """Check if image exists and is valid.

    Checks for;
        - image_id to uuid mapping
        - image file existing
        - image file matching uuid
    If all files exist and are correct, will return True.
    """
    if uid is not None:
        return _validate_uuid(uid, output_dir)

    elif image_id is not None:
        id2uid = output_dir / str(image_id)
        if id2uid.exists() and skip_existing:
            with id2uid.open(mode="r") as f:
                uid = f.readline().strip()
            return _validate_uuid(uid, output_dir)

    return False


@download_images_cli.command(name="mapillary")
def mapillary(
    *,
    skip_existing: bool = True,
    token: str | None = None,
    project: str | None = None,
):
    """Download Mapillary images to a local directory.

    Args:
        skip_existing: If true, only download missing images; otherwise overwrite.
        token: Mapillary OAuth token (if not set via MAPILLARY_TOKEN).
        project: An optional project to attach to.
    """
    from streetscapes.project import Project
    from streetscapes.sources.mapillary import MapillaryClient

    proj = Project(project or CFG.active_project)

    # TODO: perhaps move this to context in main cli?
    console.rule("Streetscapes")
    console.print(f"Active project: {proj.name}")
    console.print(f"Data home: {proj.image_dir}")

    records = proj.get_mapillary_download_records(skip_existing)

    if not records:
        logger.info("No new images to download.")
        return

    token = token or CFG.mapillary_token
    if not token:
        logger.error(
            "Error: 'mapillary_token' missing, set with `streetscapes config set "
            "mapillary_token <your token>`"
        )
        raise typer.Exit(code=1)

    mapillary = MapillaryClient(token)

    total = len(records)
    image_dir = proj.get_image_dir_for_source("mapillary")
    console.print(f"Downloading {len(records)} image(s) to {image_dir}.")

    # Add metadata to batch
    image_data = []
    downloaded = 0

    for rec in track(records, "Downloading images..."):
        (
            uid,
            image_id,
            url,
            shard,
            location,
            is_pano,
            camera_type,
        ) = rec

        # Determine the shard
        output_dir = Path(image_dir)
        shard = None
        if location is not None:
            shard = utils.get_geohash_shard_path(location)
            if output_dir is not None:
                output_dir /= shard

        if not skip_existing or not _existing_img_valid(
            uid, image_id, output_dir, skip_existing
        ):
            try:
                img_meta = mapillary.download_image(
                    url, output_dir, image_id, uid, skip_existing=skip_existing
                )
                uid = img_meta.uid
            except Exception as e:
                logger.error(e)
                continue

        tags = ["mapillary"]
        if is_pano:
            tags.append("panoramic")
        if camera_type is not None:
            tags.append(camera_type)

        image_data.append(_format_image(uid, "mapillary", str(shard), tags=tags))

        # Update the Mapillary table
        proj._con.raw_sql(f"UPDATE mapillary SET image='{uid}' WHERE id={image_id};")

        downloaded += 1

    console.print(f"Registering {len(image_data)} images...")

    proj.add_images(image_data)

    console.print(
        f"Download complete: {downloaded}/{total} images saved under "
        f"{proj.get_image_dir_for_source('mapillary')}."
    )
