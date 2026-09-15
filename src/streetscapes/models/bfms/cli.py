"""BFMS command line interface."""

from typing import Annotated, cast

import shapely
from cyclopts import Parameter

from streetscapes import CFG, utils
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.utils.logging import logger


def cli(
    *,
    image_path: str | None = None,
    model_id: str = "jinfengxie/BFMS_1014",
    run: str | None = None,
    project: str = cast("str", CFG.active_project),
    overwrite: Annotated[bool, Parameter(negative="")] = False,
    verbose: Annotated[bool, Parameter(negative="")] = False,
):
    """Segment images with BFMS.

    Args:
        image_path: Path to the images to be segmented.
            If not provided uses all downloaded images in the project.
        model_id: BFMS model ID (Huggingface format).
        run: Model run ID. Will be generated automatically if not provided.
        project: The project to use.
        overwrite: Overwrite an existing run.
        verbose: Print verbose log to the terminal. Useful for debugging models.
    """
    # Open the project
    proj = Project(project)

    model = "bfms"
    model_params = {"model_id": model_id}

    result = proj.add_run(run, model, model_params, overwrite)
    run = str(result.get("run")[0])  # type: ignore[index]

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

    # NOTE: BFMS does not support a batch mode.
    for image_idx, uid in enumerate(unprocessed, 1):
        # Read the encoded image file; decoding happens in the worker.
        path, _ = unprocessed[uid]
        request = {"image": path.read_bytes()}

        # Process the images
        logger.info(f"Segmenting image [{image_idx:>4d}/{len(unprocessed):>4d}]...")
        response = handle.remote(request).result()
        logger.debug(f"Successfully segmented image {uid}, saving instances.")

        # Save segmentation immediately
        proj.add_segmentation(
            run,
            uid,
            response.labels,
            polygons=shapely.from_wkb(response.polygons),
        )
