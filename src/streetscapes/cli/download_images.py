import os
import logging
from pathlib import Path

from cyclopts import App
from rich.progress import track

import ibis
from streetscapes import utils
from streetscapes import config
from streetscapes.cli.console import console
from streetscapes.project import Project
from streetscapes.sources.mapillary import MapillaryClient

logger = logging.getLogger(__name__)

download_images_cli = App(help="Download images from various sources.")


@download_images_cli.command(name="mapillary")
def mapillary(
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

    proj = Project(project or config.get("active_project"))

    # TODO: perhaps move this to context in main cli?
    console.rule("Streetscapes")
    console.print(f"Active project: {proj.name}")
    console.print(f"Data home: {proj.data_home}")

    records = proj.get_mapillary_download_records()

    if not records:
        logger.info("No new images to download.")
        return

    token = token or os.getenv("MAPILLARY_TOKEN")

    mapillary = MapillaryClient(token)

    total = len(records)
    image_dir = proj.get_image_path("mapillary")
    console.print(f"Downloading {total} image(s) to {image_dir}.")

    # Add metadata to batch
    image_data = {
        k: [None for _ in range(len(records))]
        for k in proj.core_tables["images"]["schema"]
    }

    downloaded = 0

    for idx, (uid, image_id, url, shard, location) in track(
        enumerate(records), "Downloading images...", total=len(records)
    ):

        # Determine the shard
        output_dir = Path(image_dir)
        shard = None
        if location is not None:
            shard = utils.get_geohash_shard_path(location)
            if output_dir is not None:
                output_dir /= shard

        # Download image
        meta = mapillary.download_image(url, output_dir, uid, skip_existing)

        meta.shard = str(shard)

        # Image registration
        image_data["uuid"][idx] = ibis.uuid(meta.iuuid).to_pyarrow()
        image_data["source"][idx] = meta.source
        image_data["shard"][idx] = meta.shard

        # Update the Mapillary table
        proj._con.raw_sql(
            f"UPDATE mapillary SET uuid='{meta.iuuid}' WHERE id={image_id};"
        )

        downloaded += 1

    console.print(f"Registering {len(image_data['uuid'])} images...")
    df = ibis.memtable(image_data)
    proj.register_images(df)

    console.print(
        f"Download complete: {downloaded}/{total} images saved under {proj.get_image_path('mapillary')}."
    )
