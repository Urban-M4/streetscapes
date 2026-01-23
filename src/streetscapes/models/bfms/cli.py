"""Command line interface for BFMS model."""

import logging
from pathlib import Path

import filetype as ft
import imageio.v3 as iio
import numpy as np
import orjson as oj

from streetscapes import config
from streetscapes.models.bfms.db import SCHEMA
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.models.bfms.db import save_segmentation

logger = logging.getLogger(__name__)


def cli(
    image_path: str,
    model_params: dict | None = None,
    overwrite: bool = False,
    bootstrap: bool = False,
):
    """Segment images with BFMS.

    Args:
        image_path: Path to an image or a directory of images.
        model_params: Optional parameters to pass to the Ray Serve deployment.
        overwrite: Whether to overwrite existing segmentations.
        bootstrap: (Re)create the model table.
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
    project = Project(config.get("active_project"))
    project.ensure_table(model_name, SCHEMA, bootstrap)

    # Determine which images need processing
    processed, unprocessed = project.get_segmentation_status(image_paths, model_name, overwrite)

    # Initialize Ray Serve handle
    handle = serve_model(model_name, **(model_params or {}))
    logger.info(f"Segmenting {len(unprocessed)} images using BFMS...")

    # Process images one by one
    for img_hash, img_path in unprocessed:

        # Extract the hashes
        image = np.asarray(iio.imread(img_path))

        # Create request for the service
        request = {"image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY)}
        response = handle.remote(request).result()
        response.hash = img_hash

        # Save segmentation immediately
        save_segmentation(project, model_params, response, processed)
