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
    model: str,
    run: str,
    params: dict | None,
    responses: DinoSAMResponse,
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

    labels = {
        "hash": [],
        "model": [],
        "run": [],
        "labels": [],
    }
    for response in responses:

        # The file name is constructed from the hash.
        seg_fname = f"{response.hash.hex()}.npz"


        # Save the segmentations.
        logger.info(f"[ Saving segmentation '{seg_fname}'...")
        instances = oj.loads(response.instances)
        np.savez_compressed(archive_path / seg_fname, instances)

        labels["hash"].append(response.hash)
        labels["model"].append(model)
        labels["run"].append(run)
        labels["labels"].append(response.labels)

    project._con.insert("labels", labels, overwrite=True)
