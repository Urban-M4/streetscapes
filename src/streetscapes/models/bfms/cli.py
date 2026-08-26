from typing import cast

import imageio.v3 as iio
import numpy as np
import orjson as oj

from streetscapes import CFG, utils
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.utils.logging import logger
from streetscapes.utils.masks import mask2poly


def cli(
    image_path: str | None = None,
    model_id: str = "jinfengxie/BFMS_1014",
    run: str | None = None,
    project: str = cast("str", CFG.active_project),
    overwrite: bool = False,
    verbose: bool = False,
):
    """Segment images with BFMS.

    Args:
        image_path: Path to the images to be segmented.
            If not provided uses all downloaded images in the project.
        model_id: BFMS model ID (Huggingface format).
        run: Model run name.
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

        # Extract the paths and open the images as NumPy arrays.
        path, _ = unprocessed[uid]
        img = np.asarray(iio.imread(path))
        request = {
            "image": oj.dumps(
                img,
                option=oj.OPT_SERIALIZE_NUMPY,
            )
        }

        # Process the images
        logger.info(f"Segmenting image [{image_idx:>4d}/{len(unprocessed):>4d}]...")
        response = handle.remote(request).result()
        logger.debug(f"Successfully segmented image {uid}, saving instances.")

        # Save the instances.
        instances = oj.loads(response.instances)
        # Save segmentation immediately
        proj.add_segmentation(
            run,
            uid,
            response.labels,
            polygons=mask2poly(np.array(instances), model="bfms", image=img,),
        )
