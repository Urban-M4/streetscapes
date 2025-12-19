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

logger = logging.getLogger(__name__)

def cli(
    image_path: str,
    overwrite: bool = False,
    model_params: dict | None = None,
):
    """CLI entry point to segment images with BFMS via Ray Serve.

    Args:
        image_path: Path to an image or a directory of images.
        overwrite: Whether to overwrite existing segmentations.
        model_params: Optional parameters to pass to the Ray Serve deployment.
    """
    # Resolve paths
    image_path = Path(image_path)
    if image_path.is_dir():
        image_paths = [p for p in image_path.glob("*.*") if ft.is_image(p)]
    else:
        image_paths = [image_path]

    if not image_paths:
        return

    # Setup project
    project = Project(config.get("active_project"))
    project.ensure_table("bfms", SCHEMA)

    # Determine which images need processing
    # processed, unprocessed = project.get_image_status(image_paths, "bfms", overwrite)
    unprocessed = image_paths

    # Initialize Ray Serve handle
    handle = serve_model("bfms", **(model_params or {}))
    logger.info(f"Segmenting {len(unprocessed)} images using BFMS...")

    # Process images one by one
    for img_path in unprocessed:
        image = np.asarray(iio.imread(img_path))

        # Create request for the service
        request = {"image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY)}
        response = handle.remote(request).result()
        mask = np.array(oj.loads(response.mask))

        # Save segmentation immediately
        # project.save_segmentation("bfms", img_path, mask)