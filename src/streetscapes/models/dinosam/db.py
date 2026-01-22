import uuid
import ibis
from pathlib import Path
import orjson as oj
import numpy as np
import ibis.expr.datatypes.core as idt

from streetscapes import utils
from streetscapes.project import Project
from streetscapes.utils import logger, ensure_dir
from streetscapes.models.dinosam.service import DinoSAMResponse


def save_segmentations(
    project: Project,
    params: dict | None,
    responses: DinoSAMResponse,
    processed: dict[bytes, uuid.UUID],
    archive_path: Path,
):
    """
    Save a segmentation.

    Args:
        project: Current project.
        params: The model parameters.
        response: A BFMS response model.
        processed: Images that have already been processed.
        archive_path: The path to the archive directory.
    """
    model_name = "dinosam"

    # Ensure that params is a dictionary.
    params = params or {}

    for response in responses:

        # Retrieve or create a UUID for this segmentation.
        seg_uuid = ibis.uuid(processed.get(response.hash, utils.uuid7()))
        seg_fname = f"{seg_uuid.to_pyarrow().as_py()}.npz"
        seg_fpath = project.get_archive_path("dinosam")

        # Save the segmentations.
        logger.info(f"[ Saving segmentation '{seg_fname}'...")
        instances = oj.loads(response.instances)
        np.savez_compressed(
            seg_fpath / seg_fname,
            labels=response.labels,
            instances=instances,
        )
