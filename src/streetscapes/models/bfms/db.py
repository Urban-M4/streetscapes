import uuid
import ibis

import orjson as oj
import numpy as np
from uuid7gen import uuid7

from streetscapes.project import Project
from streetscapes.utils import logger, ensure_dir
from streetscapes.models.bfms.service import BFMSResponse

SCHEMA = {
    # The UUID column corresponds to the name of the file
    # where the data from the model is saved.
    "uuid": ibis.dtype("!uuid"),
    "params": ibis.dtype("!binary"),
    "timestamp": ibis.dtype("!timestamp"),
}


def save_segmentation(
    project: Project,
    params: dict | None,
    response: BFMSResponse,
    processed: dict[bytes, uuid.UUID],
    shard: str | None = None,
):
    """
    Save a segmentation.

    Args:
        project: Current project.
        params: The model parameters.
        response: A BFMS response model.
        processed: Images that have already been processed.
        shard: An optional subdirectory.
    """

    model_name = "bfms"

    # Ensure that params is a dictionary.
    params = params or {}

    # Rows to be inserted into the database
    seg_rows = {k: [] for k in SCHEMA}
    m2m_rows = {k: [] for k in project.core_tables["image_model"]}

    timestamp = ibis.now()

    # ATTENTION: using UUID7 from the built-in uuid module requires Python >= 3.14.
    seg_uuid = ibis.uuid(processed.get(response.hash, uuid7()))
    seg_fname = f"{seg_uuid.to_pyarrow().as_py()}.npz"
    seg_fpath = ensure_dir(project.data_home / f"models/{model_name}/segmentations")

    # Save the segmentations.
    logger.info(f"Saving segmentation {seg_fname}...")
    np.savez_compressed(
        seg_fpath / seg_fname,
        mask=response.mask,
    )

    # Model table update.
    seg_rows["uuid"].append(seg_uuid.to_pyarrow())
    seg_rows["params"].append(oj.dumps(params))
    seg_rows["timestamp"].append(timestamp.to_pyarrow())

    # M2M table update
    m2m_rows["image_hash"].append(response.hash)
    m2m_rows["model"].append(model_name)
    m2m_rows["uuid"].append(seg_uuid.to_pyarrow())

    # Update the model database
    project._con.insert(model_name, seg_rows)
    project._con.insert("image_model", m2m_rows)
