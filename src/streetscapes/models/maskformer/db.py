import ibis
import numpy as np
import uuid
import orjson as oj
from uuid7gen import uuid7

from streetscapes.project import Project
from streetscapes.utils import logger, ensure_dir
from streetscapes.models.maskformer.service import MaskFormerResponse


SCHEMA = {
    # The UUID column corresponds to the name of the file
    # where the data from the model is saved.
    "uuid": ibis.dtype("!uuid"),
    "params": ibis.dtype("!binary"),
    "timestamp": ibis.dtype("!timestamp"),
}


def save_segmentations(
    project: Project,
    params: dict | None,
    responses: list[MaskFormerResponse],
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

    model_name = "dinosam"

    # Ensure that params is a dictionary.
    params = params or {}

    # Rows to be inserted into the database
    seg_rows = {k: [] for k in SCHEMA}
    m2m_rows = {k: [] for k in project.core_tables["image_model"]}

    timestamp = ibis.now()

    for response in responses:

        # Retrieve or create a UUID for this segmentation.
        seg_uuid = ibis.uuid(processed.get(response.uid, uuid7()))
        seg_fname = f"{seg_uuid.to_pyarrow().as_py()}.npz"
        seg_fpath = ensure_dir(project.data_home / f"models/{model_name}/segmentations")

        # Save the segmentations.
        logger.debug(f"Saving segmentation {seg_fname}...")
        instances = oj.loads(response.instances)
        np.savez_compressed(
            seg_fpath / seg_fname,
            labels=response.labels,
            instances=instances,
        )

        # Model table update.
        seg_rows["uuid"].append(seg_uuid.to_pyarrow())
        seg_rows["params"].append(oj.dumps(params))
        seg_rows["timestamp"].append(timestamp.to_pyarrow())

        # M2M table update.
        m2m_rows["image_hash"].append(response.uid)
        m2m_rows["model"].append(model_name)
        m2m_rows["uuid"].append(seg_uuid.to_pyarrow())

    # Update the model database
    project._con.insert(model_name, seg_rows)
    project._con.insert("image_model", m2m_rows)
