"""Command line interface for BFMS model."""

import logging
from itertools import batched
from typing import Annotated, cast

import imageio.v3 as iio
import numpy as np
import orjson as oj
from cyclopts import Parameter

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
    sam_model_id: str = "facebook/sam2.1-hiera-large",
    dino_model_id: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    run: str | None = None,
    project: str = cast("str", CFG.active_project),
    overwrite: Annotated[bool, Parameter(negative="")] = False,
    verbose: Annotated[bool, Parameter(negative="")] = False,
):
    """Segment images with DinoSAM.

    Args:
        prompt: The prompt to use for this model.
        image_path: Path to an image or a directory of images.
            If not provided uses all downloaded images in the project.
        batch_size: Batch size for the segmenter.
        sam_model_id: SAM model ID (Huggingface format).
        dino_model_id: Dino model ID (Huggingface format).
        box_threshold: Box threshold for Dino.
        text_threshold: Text threshold for Dino.
        overwrite: Whether to overwrite existing segmentations.
        run: Model run ID. Will be generated automatically if not provided.
        project: The project to use.
        overwrite: Overwrite an existing run.
        verbose: Print verbose log to the terminal. Useful for debugging models.
    """

    # Open the project
    proj = Project(project)

    # Save the run metadata.
    # ==================================================
    model = "dinosam"
    model_params = {
        "sam_model_id": sam_model_id,
        "dino_model_id": dino_model_id,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
    }

    result = proj.add_run(run, model, model_params, overwrite)
    run = str(result.get("run")[0])  # type: ignore[index]

    # Get all images that need to be processed.
    # ==================================================
    if image_path is not None:
        image_paths = utils.get_image_paths(image_path)
        if len(image_paths) == 0:
            logger.info(f"Nothing to process.")
            return

        uids = list(map(utils.get_image_uuid, image_paths))
    else:
        uids = proj.get_image_uuids()
    _, unprocessed = proj.get_segmentation_status(uids, run)

    if len(unprocessed) == 0:
        logger.info(f"Nothing to process.")
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
                "image": oj.dumps(
                    np.asarray(iio.imread(path)), option=oj.OPT_SERIALIZE_NUMPY
                ),
            }
            request["images"].append(img_data)  # type: ignore[attr-defined]

        # Process the images.
        logger.info(f"Segmenting batch [{batch_idx:>4d}/{len(batches):>4d}]...")
        responses = handle.remote(request).result()
        logger.debug(f"Successfully segmented {len(batch)} images, saving instances.")

        # Save the instances.
        segmentations = []
        for response in responses:
            instances = oj.loads(response.instances)
            instances = np.array(instances) # turned 3-level nested list into 3D array
            segmentations.append(
                {
                    "run": run,
                    "image": response.uid,
                    "labels": response.labels,
                    "polygons": mask2poly(instances, model="dinosam"),
                }
            )

        # Update the segmentation table.
        proj.add_segmentations(segmentations, overwrite)
