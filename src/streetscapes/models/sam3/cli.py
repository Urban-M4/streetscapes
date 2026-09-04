"""Command line interface for BFMS model."""

import importlib.util
import logging
from itertools import batched
from typing import Annotated, cast

import imageio.v3 as iio
import numpy as np
from cyclopts import Parameter
from ray import cloudpickle

from streetscapes import CFG, utils
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.utils.masks import mask2poly

logger = logging.getLogger(__name__)


def cli(
    prompt: str,
    /,
    *,
    image_path: str | None = None,
    batch_size: int = 10,
    confidence: float = 0.25,
    quantisation: str | None = None,
    run: str | None = None,
    project: str = cast("str", CFG.active_project),
    overwrite: Annotated[bool, Parameter(negative="")] = False,
    verbose: Annotated[bool, Parameter(negative="")] = False,
):
    """Segment images with SAM3.

    Note: The SAM3 model weights cannot be downloaded from HuggingFace without
    authentication. Instead download the SAM3 `.pt` file after signing up at:
    https://huggingface.co/facebook/sam3
    Set the absolute path to the downloaded SAM3 `.pt` file in the
    streetscapes config:
    > streetscapes config set sam3_model_path "/abs/path/to/file.pt".

    Args:
        prompt: The prompt to use for this model (e.g. "tree,bench,sign").
        image_path: Path to an image or a directory of images.
            If not provided uses all downloaded images in the project.
        batch_size: Batch size for the segmenter.
        device: Specify a device to run the model on.
        confidence: Confidence threshold for accepting segmentations.
        quantisation: Quantisation level. Possible values are `FP16` (faster inference)
            or `FP32`. `None` means that the default value will be used.
        overwrite: Whether to overwrite existing segmentations.
        run: Model run ID. Will be generated automatically if not provided.
        project: The project to use.
        overwrite: Overwrite an existing run.
        verbose: Print verbose log to the terminal. Useful for debugging models.
    """
    if importlib.util.find_spec("ultralytics") is None:
        msg = (
            "SAM3 requires extra dependencies. "
            "Install these with `pip install streetscapes[sam3]` or"
            "`uv sync --all-extras`."
        )
        raise ImportError(msg)

    if CFG.sam3_model_path is None:
        raise ValueError(
            "No SAM3 model weights configured. Configure the 'sam3_model_path' entry"
            " in the streetscapes config."
        )

    # Open the project
    proj = Project(project)

    # Save the run metadata.
    # ==================================================
    model = "sam3"
    model_params = {
        "weights": str(CFG.sam3_model_path),
        "confidence": confidence,
        "quantisation": quantisation,
    }

    result = proj.add_run(run, model, model_params | {"prompt": prompt}, overwrite)
    run = str(result.get("run")[0])  # type: ignore[index]

    # Get all images that need to be processed.
    # ==================================================
    if image_path is not None:
        image_paths = utils.get_image_paths(image_path)
        if len(image_paths) == 0:
            logger.info("Nothing to process.")
            return

        uids = list(map(utils.get_image_uuid, image_paths))
    else:
        uids = proj.get_image_uuids()
    _, unprocessed = proj.get_segmentation_status(uids, run)

    if len(unprocessed) == 0:
        logger.info("Nothing to process.")
        return

    handle = serve_model(model, verbose, **model_params)
    logger.info(f"Segmenting {len(unprocessed)} images using {model}...")
    batches = list(batched(unprocessed, batch_size))
    for batch_idx, batch in enumerate(batches, 1):
        # Extract the paths and open the images as NumPy arrays.
        request = {
            "images": [],
            "prompt": prompt,
        }
        for uid in batch:
            path, _ = unprocessed[uid]
            img_data = {
                "uid": uid,
                "image": cloudpickle.dumps(np.asarray(iio.imread(path))),
            }
            request["images"].append(img_data)  # type: ignore[attr-defined]

        # Process the images.
        logger.info(f"Segmenting batch [{batch_idx:>4d}/{len(batches):>4d}]...")
        responses = handle.remote(request).result()
        logger.debug(f"Successfully segmented {len(batch)} images, saving instances.")

        # Save the instances.
        segmentations = []
        for response in responses:
            instances = cloudpickle.loads(response.instances)
            segmentations.append(
                {
                    "run": run,
                    "image": response.uid,
                    "labels": response.labels,
                    "confidences": response.confidences,
                    "polygons": mask2poly(instances, model="dinosam"),
                }
            )

        # Update the segmentation table.
        proj.add_segmentations(segmentations, overwrite)
