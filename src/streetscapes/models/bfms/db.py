import uuid
import ibis

import numpy as np

from streetscapes.project import Project
from streetscapes.utils import logger

SCHEMA = {
    "image_hash": ibis.dtype("!binary"),
    "params": ibis.dtype("!json"),
    "segmentation": ibis.dtype("!str"),
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
    seg_rows = {k: [] for k in SCHEMA()}
    m2m_rows = {k: [] for k in project.core_tables["image_model"]}

    timestamp = ibis.now()

    for segmentation in segmentations:
        seg_uuid = ibis.uuid(processed.get(segmentation.image_hash, uuid.uuid7()))
        logger.info(f"seg_uuid: {seg_uuid} | hash: {segmentation.image_hash}")
        seg_fpath = project.data_home / f"{seg_uuid}.npz"

        # Save the segmentations.
        np.savez(seg_fpath, instances=segmentation.instances, masks=segmentation.masks)

        # Model table update
        seg_rows["uuid"].append(seg_uuid.to_pyarrow())
        seg_rows["params"].append(params)
        seg_rows["timestamp"].append(timestamp.to_pyarrow())

        # M2M table update
        m2m_rows["image_hash"].append(segmentation.image_hash)
        m2m_rows["model"].append("bfms")
        m2m_rows["uuid"].append(seg_uuid.to_pyarrow())

    # Update the model database
    project.con.insert("bfms", seg_rows)
    project.con.insert("image_model", m2m_rows)
