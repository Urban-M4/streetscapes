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

    # Open the project
    project = Project(config.get("active_project"))

    # Determine which images need processing
    status = project.get_segmentation_status(collection, model_name, run)

    if status is None:
        logger.info(f"Nothing to segment.")
        return

    processed, unprocessed = status

    # Initialize Ray Serve handle
    handle = serve_model(model_name, **model_params)
    logger.info(f"Segmenting {len(unprocessed)} images using {model_name}...")

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
    model_params = {
        "sam_model_id": sam_model_id,
        "dino_model_id": dino_model_id,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
    }
    seg_data = {
        "collection": collection,
        "model": model_name,
        "run": run,
        "archive": ibis.uuid(run_uid).to_pyarrow(),
        "params": oj.dumps(model_params),
    }
    project.update_table("segmentations", seg_data)

    logger.info(f"Segmenting {len(unprocessed)} images...")

    for batch in batched(unprocessed.items(), batch_size):

        # Extract the hashes
        uids = [b[0] for b in batch]
        images = [np.asarray(iio.imread(b[1])) for b in batch]

        # Process the images
        data = {
            "images": [
                {
                    "hash": h,
                    "image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY),
                }
                for h, image in zip(uids, images)
            ],
            "prompt": prompt,
        }
        responses = handle.remote(data).result()

        logger.info(f"Successfully segmented {len(images)} images, saving instances.")

        for idx, response in enumerate(responses):

            label_data["image"][idx] = ibis.uuid(response.uuid).to_pyarrow()
            label_data["model"][idx] = model_name
            label_data["run"][idx] = run
            label_data["labels"][idx] = response.labels

            # The file name is constructed from the UUID.
            seg_fname = f"{response.uuid}.npz"

            # Save the segmentations.
            logger.info(f"[ Saving segmentation '{seg_fname}'...")
            instances = oj.loads(response.instances)

            project.save_segmentation(archive_path, instances)


    # Add the labels to the database
    project.update_table('labels', label_data)
