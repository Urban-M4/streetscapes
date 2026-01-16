from pathlib import Path
from itertools import batched
import orjson as oj
import numpy as np
import imageio as iio
import filetype as ft

from streetscapes.utils import logger
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.models.maskformer.db import SCHEMA, save_segmentations
from streetscapes.models.maskformer.model import MaskFormer


def cli(
    image_path: str,
    labels: list[str] | None = None,
    batch_size: int = 10,
    model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_threshold: float = 0.8,
    fuse_labels: list[str] | None = None,
    overwrite: bool = False,
    bootstrap: bool = False,
    project: str = "streetscapes",
):
    """Segment images with MaskFormer.

    Args:
        image_path: Path to the images to be segmented.
        labels: Labels to focus on.
        batch_size: Batch size for the segmentation model.
        model_id: Mask2Former model to load.
        threshold: The probability score threshold to keep predicted instance masks.
        mask_threshold: Threshold to use when turning the predicted masks into binary values.
        overlap_threshold: The overlap mask area threshold to merge or discard small
            disconnected parts within each binary instance mask.
        fuse_labels: The labels in this state will have all their instances fused together.
        overwrite: Overwrite existing segmentations.
        bootstrap: (Re)create the model table.
        project: The project to use for saving (meta)data.
    """

    if fuse_labels is None or len(fuse_labels) < 2:
        # Fusing a single label makes no sense...
        fuse_labels = []

    model_name = "maskformer"

    model_params = {
        "model_id": model_id,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "overlap_mask_area_threshold": overlap_threshold,
        "labels_to_fuse": fuse_labels,
    }

    if image_path is not None:
        image_path = Path(image_path)

    if image_path.is_dir():
        image_path = [
            im_path for im_path in image_path.glob("*.*") if ft.is_image(im_path)
        ]

    # Open the project
    project = Project(project)
    project.ensure_table(model_name, SCHEMA, bootstrap)

    if model_params is None:
        model_params = {}

    if labels is None:
        labels = {l: None for l in MaskFormer.id_to_label.values()}

    (processed, unprocessed) = project.get_image_status(
        image_path, model_name, overwrite
    )

    handle = serve_model(model_name, **model_params)
    logger.info(f"Segmenting {len(unprocessed)} images using {model_name}...")

    for entries in batched(unprocessed, batch_size):

        # Extract the hashes
        hashes = [e[0] for e in entries]
        images = [np.asarray(iio.imread(e[1])) for e in entries]

        logger.info(f"Segmenting {len(images)} images.")
        # Process the images
        data = {
            "images": [
                {
                    "hash": h,
                    "image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY),
                }
                for h, image in zip(hashes, images)
            ],
            "labels": labels,
        }
        responses = handle.remote(data).result()

        logger.info(f"Successfully segmented {len(images)} images, saving to database.")

        # Store the segmentations and their metadata
        save_segmentations(project, model_params, responses, processed)
