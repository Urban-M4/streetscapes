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
        "segmentation": ibis.dtype("!str"),
        "timestamp": ibis.dtype("!timestamp"),
    }


def save_segmentations(
    project: Project,
    params: dict,
    segmentations: list,
    overwrite: set[bytes],
):
    """
    Save a list of segmentations.

    Args:
        project: Current project.
        params: The model parameters.
        segmentations: The list of segmentations.
        overwrite: Processed images whose segmentations should be overwritten.
    """

    # Rows to be inserted into the database
    rows = {k: [] for k in get_db_schema()}

    timestamp = ibis.now()

    for segmentation in segmentations:

        if segmentation.image_hash in overwrite:
            seg_uuid = (
                project.con.table("maskformer")
                .select("uuid")
                .to_pyarrow()
                .to_pydict()["uuid"][0]
            )
        else:
            seg_uuid = uuid.uuid4()

        seg_fpath = project.get_output_dir("maskformer", True) / f"{seg_uuid}.npz"

        # Save the segmentations.
        np.savez(seg_fpath, instances=segmentation.instances, masks=segmentation.masks)

        # Update the row dictionary
        rows["image_hash"].append(segmentation.image_hash)
        rows["params"].append(params)
        rows["segmentation"].append(str(seg_fpath))
        rows["timestamp"].append(timestamp.to_pyarrow())

    # Update the database
    project.con.insert("maskformer", rows)


def get_unprocessed_images(
    project: Project,
    image_paths: list[Path],
    overwrite: bool = False,
) -> list[tuple[bytes, np.ndarray]]:
    """
    Filter out processed images. Using the sha256 hash as the unique image ID.

    Args:
        project: Project object.
        image_paths: Image paths to process.
        overwrite: Overwrite processed images.

    Returns:
        A list of paths to unprocessed image.
    """

    hashes = {
        sha256(np.asarray(iio.imread(path))).digest(): path for path in image_paths
    }

    t = project.con.table("maskformer")
    processed = set(
        t.filter(t.image_hash.isin(list(hashes.keys())))
        .select("image_hash")
        .to_pyarrow()
        .to_pydict()["image_hash"]
    )
    unprocessed = [(k, v) for k, v in hashes.items() if k not in processed]
    if not overwrite:
        processed = set()
    return unprocessed, processed


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
    project.ensure_table("maskformer", get_db_schema(), recreate=True)

    if model_params is None:
        model_params = {}

    if labels is None:
        labels = {l: None for l in MaskFormer.id_to_label.values()}

    to_process, to_overwrite = get_unprocessed_images(project, image_paths, overwrite)

    handle = serve_model("maskformer", **model_params)

    for entries in batched(to_process, batch_size):

        hashes = [e[0] for e in entries]
        images = [np.asarray(iio.imread(e[1])) for e in entries]

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
        save_segmentations(project, model_params, response, to_overwrite)
