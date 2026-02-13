from itertools import batched
import imageio.v3 as iio
import numpy as np
import orjson as oj
from streetscapes import utils
from streetscapes.utils.logging import logger
from streetscapes.project import Project
from streetscapes.serve.server import serve_model


def cli(
    image_path: str,
    run: str | None = None,
    project: str = "streetscapes",
    overwrite: bool = False,
):
    """
    Segment images with BFMS.

    Args:
        image_path: Path to the images to be segmented.
        run: Model run ID.
        project: The project to use.
        overwrite: Overwrite an existing run.
    """

    # Open the project
    proj = Project(project)

    # Save the run metadata.
    # ==================================================
    model = "bfms"
    model_params = {}
    if run is None:
        run = utils.uuid7(as_str=True)

    proj.add_run(run, model, model_params, overwrite)

    # Get all images that need to be processed.
    # ==================================================
    image_paths = utils.get_image_paths(image_path)
    if len(image_paths) == 0:
        logger.info(f"Nothing to process.")
        return

    uids = list(map(utils.get_image_uuid, image_paths))
    processed, unprocessed = proj.get_segmentation_status(uids, run)

    if len(unprocessed) == 0:
        logger.info(f"Nothing to process.")
        return

    # Create the archive directory.
    # ==================================================
    archive_dir = utils.ensure_dir(
        proj.get_archive_dir_for_model(
            model,
            create=True,
        )
        / str(run)
    )

    # Segment the images and save the segmentations.
    # ==================================================
    # Ray Serve handle.
    handle = serve_model(model)
    logger.info(f"Segmenting {len(unprocessed)} images using {model}...")

    # NOTE: BFMS does not support a batch mode.
    for image_idx, uid in enumerate(unprocessed, 1):

        # Extract the paths and open the images as NumPy arrays.
        path, source = unprocessed[uid]
        request = {
            "image": oj.dumps(
                np.asarray(iio.imread(path)),
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
        proj.add_segmentation(run, uid, response.labels)
