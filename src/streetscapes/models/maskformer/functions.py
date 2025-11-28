from pathlib import Path

import uuid

# TODO: uuid7gen can be removed if we ever move
# to Python >=3.14 as the default since the built-in
# uuid module provides uuid7 support.
from uuid7gen import uuid7
import orjson as oj

import numpy as np
import ibis

from streetscapes.project import Project
from streetscapes.serve import serve_model
from streetscapes.models.maskformer import MaskFormer


def get_db_schema() -> dict:
    """
    Database schema for the Maskformer table.

    NOTE: '!' in front of the type means 'non-nullable':
    https://ibis-project.org/reference/datatypes#parameters

    Returns:
        A dictionary describing the database schema for Maskformer segmentations.
    """

    return {
        "id": ibis.dtype("!uuid"),
        "params": ibis.dtype("!json"),
        "segmentation": ibis.dtype("!str"),
        "timestamp": ibis.dtype("!timestamp"),
    }


def save_segmentations(
    project: Project,
    params: dict,
    segmentations: list,
):
    """
    Save a list of segmentations.

    Args:
        project: Current project.
        params: The model parameters.
        segmentations: The list of segmentations.
    """

    # Rows to be inserted into the database
    rows = {k: [] for k in get_db_schema()}

    timestamp = ibis.now()

    for segmentation in segmentations:
        seg_id = uuid.uuid4()
        seg_fpath = project.get_output_dir("maskformer", True) / f"{seg_id}.npz"

        # Save the segmentations.
        np.savez(seg_fpath, segmentation=segmentation)

        params = oj.dumps(params)

        # Update the row dictionary
        rows["id"].append(ibis.uuid(uuid7(timestamp_ms=1e-3)).to_pyarrow())
        rows["params"].append(params)
        rows["segmentation"].append(str(seg_fpath))
        rows["timestamp"].append(timestamp.to_pyarrow())

    # Update the database
    project.con.insert("maskformer", rows)


def segment_images(
    image_path: str | Path,
    labels: dict | None = None,
    batch_size: int = 10,
    model_params: dict | None = None,
    overwrite: bool = False,
    project: str | None = None,
):

    project = Project(project)
    project.ensure_table("maskformer", get_db_schema(), replace=True)

    image_path = Path(image_path)

    if model_params is None:
        model_params = {}

    if labels is None:
        labels = {l: None for l in MaskFormer.id_to_label.values()}

    handle = serve_model("maskformer", **model_params)

    data = {
        "image_path": image_path,
        "labels": labels,
        "batch_size": batch_size,
    }
    response = handle.remote(data).result()

    # Get the dedicated Maskformer table
    save_segmentations(project, model_params, response)
