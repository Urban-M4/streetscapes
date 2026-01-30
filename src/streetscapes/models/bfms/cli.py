"""Command line interface for BFMS model."""

from pathlib import Path
import ibis

import filetype as ft
import imageio.v3 as iio
import numpy as np
import orjson as oj

from streetscapes import config
from streetscapes import utils
from streetscapes.utils.logging import logger
from streetscapes.project import Project
from streetscapes.serve.server import serve_model


def cli(
    image_path: str,
    collection: str,
    run: str = str(utils.iso_timestamp()),
    overwrite: bool = False,
    project: str | None = None,
):
    """Segment images with BFMS.

    Args:
        image_path: Path to an image or a directory of images.
        collection: A named image subset.
        run: A run identifier.
        overwrite: Whether to overwrite existing segmentations.
        project: An optional project to attach to.
    """

    # Resolve paths
    image_path = Path(image_path)
    if image_path.is_dir():
        image_paths = [p for p in image_path.glob("*.*") if ft.is_image(p)]
    else:
        image_paths = [image_path] if ft.is_image(image_path) else []

    if not image_paths:
        return

    model_name = "bfms"

    # Open the project
    project = Project(project or config.get("active_project"))

    # Determine which images need processing
    status = project.get_segmentation_status(collection, model_name, run)

    if status is None:
        logger.info(f"Nothing to segment.")
        return

    processed, unprocessed = status

    # Initialize Ray Serve handle
    handle = serve_model(model_name)
    logger.info(f"Segmenting {len(unprocessed)} images using BFMS...")

    # Rows to be inserted into the database
    seg_data = {k: [] for k in Project.core_tables["segmentations"]["schema"]}
    run_uid = project.get_segmentation_run_uid(collection, model_name, run, create=True)

    # Update the segmentation database
    project._con.insert("segmentations", seg_data)
    archive_path = utils.ensure_dir(
        project.get_archive_path(model_name, create=True) / str(run_uid)
    )

    label_data = {
        k: [None for _ in range(len(unprocessed))]
        for k in project.core_tables["labels"]["schema"]
    }

    # Add the segmentation run to the databse
    model_params = {}
    seg_data = {
        "collection": collection,
        "model": model_name,
        "run": run,
        "archive": ibis.uuid(run_uid).to_pyarrow(),
        "params": oj.dumps(model_params),
    }
    project.update_table("segmentations", seg_data)

    logger.info(f"Segmenting {len(unprocessed)} images...")
    # Process images one by one
    for idx, (uuid, (path, shard)) in enumerate(unprocessed.items()):

        # Extract the hashes
        image = np.asarray(iio.imread(path))

        label_data["image"][idx] = ibis.uuid(response.uuid).to_pyarrow()
        label_data["model"][idx] = model_name
        label_data["run"][idx] = run
        label_data["labels"][idx] = response.labels

        # Create request for the service
        request = {"image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY)}
        response = handle.remote(request).result()
        response.uid = uuid

        # The file name is constructed from the UUID.
        seg_fname = f"{response.uuid}.npz"

        # Save the segmentations.
        logger.info(f"Saving segmentation '{seg_fname}'...")
        instances = oj.loads(response.segmentation)

        # Save segmentation immediately
        project.save_segmentation(archive_path / seg_fname, instances, overwrite)

    # Add the labels to the database
    project.update_table("labels", label_data, overwrite)
