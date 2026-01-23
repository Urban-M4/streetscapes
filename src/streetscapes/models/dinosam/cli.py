"""Command line interface for BFMS model."""

import logging
from pathlib import Path
from itertools import batched
import filetype as ft
import imageio.v3 as iio
import numpy as np
import orjson as oj
import ibis

from streetscapes import utils
from streetscapes import config
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.models.dinosam.db import save_segmentations

logger = logging.getLogger(__name__)


def cli(
    image_path: str,
    prompt: str,
    collection: str,
    run: str = str(utils.iso_timestamp()),
    batch_size: int = 10,
    sam_model_id: str = "facebook/sam2.1-hiera-large",
    dino_model_id: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    overwrite: bool = False,
    bootstrap: bool = False,
):
    """Segment images with DinoSAM.

    Args:
        image_path: Path to an image or a directory of images.
        prompt: The prompt to use for this model.
        collection: A named image subset.
        run: A run identifier.
        batch_size: Batch size for the segmenter.
        sam_model_id: SAM model ID (Huggingface format).
        dino_model_id: Dino model ID (Huggingface format).
        box_threshold: Box threshold for Dino.
        text_threshold: Text threshold for Dino.
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

    model_name = "dinosam"

    model_params = {
        "sam_model_id": sam_model_id,
        "dino_model_id": dino_model_id,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
    }

    # Open the project
    project = Project(config.get("active_project"))

    # Determine which images need processing
    processed, unprocessed = project.get_segmentation_status(
        image_paths,
        model_name,
        overwrite,
    )

    # Initialize Ray Serve handle
    handle = serve_model(model_name, **model_params)
    logger.info(f"Segmenting {len(unprocessed)} images using DinoSAM...")

    # Rows to be inserted into the database
    seg_rows = {k: [] for k in Project.core_tables["segmentations"]["schema"]}
    seg_uuid = project.get_archive_uuid(collection, model_name, run, create=True)

    # Segmentation table update.
    seg_rows["collection"].append(collection)
    seg_rows["model"].append(model_name)
    seg_rows["run"].append(run)
    seg_rows["archive"].append(ibis.uuid(seg_uuid).to_pyarrow())
    seg_rows["params"].append(oj.dumps(model_params))

    # Update the segmentation database
    project._con.insert("segmentations", seg_rows)
    archive_path = utils.ensure_dir(project.get_archive_path(model_name, create=True) / seg_uuid)

    for entries in batched(unprocessed, batch_size):

        # Extract the hashes
        hashes = [e[0] for e in entries]
        images = [np.asarray(iio.imread(e[1])) for e in entries]

        # Process the images
        data = {
            "images": [
                {
                    "hash": h,
                    "image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY),
                }
                for h, image in zip(hashes, images)
            ],
            "prompt": prompt,
        }
        responses = handle.remote(data).result().model_dump()

        logger.info(f"Successfully segmented {len(images)} images, saving instances.")

        # Store the segmentations and their metadata
        save_segmentations(project, model_params, responses, processed, archive_path)
