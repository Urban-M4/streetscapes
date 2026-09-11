"""Image download CLI.

usage:
> streetscapes download-images --help
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from cyclopts import App
from rich.progress import track

from streetscapes import CFG, utils
from streetscapes.cli.console import console
from streetscapes.project import _format_image

if TYPE_CHECKING:
    import uuid

    from streetscapes.project import Project
    from streetscapes.utils.metadata import ImageMeta

logger = logging.getLogger(__name__)

download_images_cli = App(help="Download images from various sources.")


class SourceClient(Protocol):
    """The part of a source's client that the download command relies on."""

    def download_image(
        self,
        url: str,
        output_dir: str | Path,
        image_id: int | str | None,
        uid: "uuid.UUID | None" = None,
        skip_existing: bool = True,
    ) -> "ImageMeta":
        """Download an image and return its metadata."""
        ...


def _validate_uuid(uid: str, output_dir: Path) -> bool:
    img_file = (output_dir / uid).with_suffix(".jpg")
    if img_file.exists() and uid == utils.get_image_uuid(img_file):
        return True
    return False


def _existing_img_valid(
    uid: str | None,
    image_id: int | str | None,
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


def _download_images(
    proj: "Project",
    client: SourceClient,
    source: str,
    records: list[tuple[Any, ...]],
    skip_existing: bool,
):
    """Download a source's images and register them with the project.

    Args:
        proj: The project to download for.
        client: The source's client.
        source: The source name (e.g. 'mapillary', 'kartaview').
        records: The records to download, as returned by
            `Project.get_download_records`.
        skip_existing: If true, only download missing images.
    """
    total = len(records)
    image_dir = proj.get_image_dir_for_source(source)
    console.print(f"Downloading {total} image(s) to {image_dir}.")

    # Add metadata to batch
    image_data = []
    downloaded = 0

    for rec in track(records, "Downloading images..."):
        (
            uid,
            image_id,
            url,
            _shard,
            location,
            is_pano,
        ) = rec

        # Determine the shard
        output_dir = Path(image_dir)
        shard = None
        if location is not None:
            shard = str(utils.get_geohash_shard_path(location))
            output_dir /= shard

        if not skip_existing or not _existing_img_valid(
            uid, image_id, output_dir, skip_existing
        ):
            try:
                img_meta = client.download_image(
                    url, output_dir, image_id, uid, skip_existing=skip_existing
                )
                uid = img_meta.uid
            except Exception as e:
                logger.error(e)
                continue

        tags = [source]
        if is_pano:
            tags.append("panoramic")

        image_data.append(_format_image(uid, source, shard, tags=tags))

        # Update the source table. The ID is quoted because Panoramax identifies
        # its pictures by UUID; the numeric IDs of the other sources still cast.
        proj._con.raw_sql(f"UPDATE {source} SET image='{uid}' WHERE id='{image_id}';")

        downloaded += 1

    console.print(f"Registering {len(image_data)} images...")

    proj.add_images(image_data)

    console.print(
        f"Download complete: {downloaded}/{total} images saved under {image_dir}."
    )


def _show_project(proj: "Project"):
    # TODO: perhaps move this to context in main cli?
    console.rule("Streetscapes")
    console.print(f"Active project: {proj.name}")
    console.print(f"Data home: {proj.image_dir}")


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
    _show_project(proj)

    records = proj.get_download_records("mapillary", "thumb_2048_url", skip_existing)

    if not records:
        logger.info("No new images to download.")
        return

    token = token or CFG.mapillary_token
    if not token:
        logger.error(
            "Error: 'mapillary_token' missing, set with `streetscapes config set "
            "mapillary_token <your token>`"
        )
        raise SystemExit(1)

    _download_images(proj, MapillaryClient(token), "mapillary", records, skip_existing)


@download_images_cli.command(name="kartaview")
def kartaview(
    *,
    skip_existing: bool = True,
    project: str | None = None,
):
    """Download KartaView images to a local directory.

    Args:
        skip_existing: If true, only download missing images; otherwise overwrite.
        project: An optional project to attach to.
    """
    from streetscapes.project import Project
    from streetscapes.sources.kartaview import KartaViewClient

    proj = Project(project or CFG.active_project)
    _show_project(proj)

    records = proj.get_download_records("kartaview", "image_url", skip_existing)

    if not records:
        logger.info("No new images to download.")
        return

    _download_images(proj, KartaViewClient(), "kartaview", records, skip_existing)


@download_images_cli.command(name="panoramax")
def panoramax(
    *,
    skip_existing: bool = True,
    project: str | None = None,
):
    """Download Panoramax images to a local directory.

    Images are downloaded from the instance hosting them, which is recorded when
    the metadata is fetched, so no instance needs to be given here.

    Args:
        skip_existing: If true, only download missing images; otherwise overwrite.
        project: An optional project to attach to.
    """
    from streetscapes.project import Project
    from streetscapes.sources.panoramax import PanoramaxClient

    proj = Project(project or CFG.active_project)
    _show_project(proj)

    records = proj.get_download_records("panoramax", "image_url", skip_existing)

    if not records:
        logger.info("No new images to download.")
        return

    _download_images(proj, PanoramaxClient(), "panoramax", records, skip_existing)
