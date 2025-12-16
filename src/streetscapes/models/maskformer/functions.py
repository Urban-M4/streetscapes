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
        "image_hash": ibis.dtype("!binary"),
        "params": ibis.dtype("!json"),
        "uuid": ibis.dtype("!str"),
        "timestamp": ibis.dtype("!timestamp"),
    }


def save_segmentations(
    project: Project,
    hashes: list[bytes],
    params: dict,
    segmentations: list,
    uuids: set[bytes],
):
    """
    Save a list of segmentations.

    Args:
        project: Current project.
        hashes: SHA265 hashes of the processed images.
        params: The model parameters.
        segmentations: The list of segmentations.
        uuids: UUIDs of the associated segmentations.
    """

    # Rows to be inserted into the database
    rows = {k: [] for k in get_db_schema()}

    timestamp = ibis.now()

    for segmentation in segmentations:

        if segmentation.image_hash in uuids:
            seg_uuid = uuids[segmentation.image_hash]
        else:
            seg_uuid = uuid.uuid4()

        seg_fpath = f"{seg_uuid}.npz"

        # Save the segmentations.
        np.savez(seg_fpath, instances=segmentation.instances, masks=segmentation.masks)

        # Update the row dictionary
        rows["image_hash"].append(ibis.uuid(uuid7()).to_pyarrow())
        rows["params"].append(params)
        rows["uuid"].append(ibis.uuid(seg_uuid))
        rows["timestamp"].append(timestamp.to_pyarrow())

    # Update the model database
    project.con.insert("maskformer", rows)


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

    handle = serve_model("maskformer", **model_params)

    for entries in batched(unprocessed, batch_size):

        # Extract the hashes
        hashes = [e[0] for e in entries]
        images = [np.asarray(iio.imread(e[1])) for e in entries]
        segmentations = (
            project.con.table("maskformer")
            .select("uuid", "segmentation")
            .to_pyarrow()
            .to_pydict()
        )
        existing = {u: s for u, s in zip(existing["uuid"], existing["segmentation"])}

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
        save_segmentations(project, hashes, model_params, response, existing)
