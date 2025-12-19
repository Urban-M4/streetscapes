from pathlib import Path
from itertools import batched

import uuid

# TODO: uuid7gen can be removed if we ever move
# to Python >=3.14 as the default since the built-in
# uuid module provides uuid7 support.
from uuid7gen import uuid7
import orjson as oj

from hashlib import sha256

import numpy as np
import ibis

import imageio as iio

from streetscapes.project import Project
from streetscapes.serve import serve_model
from streetscapes.models.maskformer import MaskFormer


def get_db_schema() -> dict:
    """
    Database schema for the Maskformer table.

    Returns:
        A dictionary describing the database schema for Maskformer segmentations.
    """

    return {
        "params": ibis.dtype("!json"),
        "uuid": ibis.dtype("!str"),
        "timestamp": ibis.dtype("!timestamp"),
    }


def save_segmentations(
    project: Project,
    params: dict,
    segmentations: list,
    processed: dict[bytes, uuid.UUID],
):
    """
    Save a list of segmentations.

    Args:
        project: Current project.
        params: The model parameters.
        segmentations: The list of segmentations.
        processed: Images that have already been processed.
    """

    # Rows to be inserted into the database
    seg_rows = {k: [] for k in get_db_schema()}
    lut_rows = {k: [] for k in project.core_tables['image_model']}

    timestamp = ibis.now()

    for segmentation in segmentations:

        seg_uuid = ibis.uuid((
            processed[segmentation.image_hash]
            if segmentation.image_hash in processed
            else uuid7()
        )

        seg_fpath = project.data_home / f"{seg_uuid}.npz"

        # Save the segmentations.
        np.savez(seg_fpath, instances=segmentation.instances, masks=segmentation.masks)

        # Model table
        seg_rows["uuid"].append(seg_uuid)
        seg_rows["params"].append(params)
        seg_rows["timestamp"].append(timestamp.to_pyarrow())

        # Intermediate lookup table
        lut_rows['image_hash'].append(segmentation.image_hash)
        lut_rows['model'].append("maskformer")
        lut_rows['uuid'].append(seg_uuid)

    # Update the model database
    project.con.insert("maskformer", seg_rows)
    project.con.insert("image_model", lut_rows)


def segment_images(
    image_paths: str | Path,
    labels: dict | None = None,
    batch_size: int = 10,
    model_params: dict | None = None,
    overwrite: bool = False,
    project: str | None = None,
):
    """
    Segment a collection of images.

    Args:
        image_paths: A list of paths to individual images.
        labels: Labels to focus on.
        batch_size: Batch size to use.
        model_params: Model parameters to pass to the model instance.
        overwrite: Overwrite processed images.
        project: The project to open.
    """

    project = Project(project)
    # TODO: Remove the "replace" argument when the API is stabilised.
    project.ensure_table("maskformer", get_db_schema(), overwrite=True)

    if model_params is None:
        model_params = {}

    if labels is None:
        labels = {l: None for l in MaskFormer.id_to_label.values()}

    (processed, unprocessed) = project.get_unprocessed_images(
        image_paths, "maskformer", overwrite
    )
    if overwrite:
        processed = {}

    handle = serve_model("maskformer", **model_params)

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
            "labels": labels,
        }
        response = handle.remote(data).result()

        # Store the segmentations and their metadata
        save_segmentations(project, model_params, response, processed)
