from typing import Optional, cast

import imageio.v3 as iio
import numpy as np
import orjson as oj

from streetscapes import CFG, utils
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.utils.logging import logger
from streetscapes.utils.masks import mask2poly


def cli(
    image_path: Optional[str] = None,
    model_id: str = "jinfengxie/BFMS_1014",
    run: Optional[str] = None,
    project: str = cast("str", CFG.active_project),
    overwrite: bool = False,
):
    """
    Segment images with BFMS.

    Args:
        image_path: Path to the images to be segmented.
            If not provided uses all downloaded images in the project.
        model_id: BFMS model ID (Huggingface format).
        run: Model run name.
        project: The project to use.
        overwrite: Overwrite an existing run.
    """

    # Open the project
    proj = Project(project)

    model = "bfms"
    model_params = {"model_id": model_id}
    if run is None:
        run = utils.uuid7(as_str=True)

    proj.add_run(run, model, model_params, overwrite)

    if image_path is not None:
        image_paths = utils.get_image_paths(image_path)
        if len(image_paths) == 0:
            logger.info(f"Nothing to process.")
            return

        uids = list(map(utils.get_image_uuid, image_paths))
    else:
        uids = proj.get_image_uuids()
    processed, unprocessed = proj.get_segmentation_status(uids, run)

    if len(unprocessed) == 0:
        logger.info(f"Nothing to process.")
        return

    archive_dir = utils.ensure_dir(
        proj.get_archive_dir_for_model(
            model,
            create=True,
        )
        / str(run)
    )

    handle = serve_model(model, **model_params)
    logger.info(f"Segmenting {len(unprocessed)} images using {model}...")

    # NOTE: BFMS does not support a batch mode.
    for image_idx, uid in enumerate(unprocessed, 1):

        # Extract the paths and open the images as NumPy arrays.
        path, source = unprocessed[uid]
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
        sub = path.relative_to(proj.get_image_dir_for_source(source))
        instances = oj.loads(response.instances)
        utils.save_instances(archive_dir / sub, instances)
        # Save segmentation immediately
        proj.add_segmentation(
            run,
            uid,
            response.labels,
            polygons=mask2poly(np.array(instances), model="bfms", image=img,),
        )
