from itertools import batched
from typing import cast

import imageio as iio
import numpy as np
import orjson as oj

from streetscapes import CFG, utils
from streetscapes.models.maskformer.model import MaskFormer
from streetscapes.project import Project
from streetscapes.serve.server import serve_model
from streetscapes.utils import logger
from streetscapes.utils.masks import mask2poly


def cli(
    image_path: str | None = None,
    labels: list[str] | None = None,
    batch_size: int = 10,
    model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_threshold: float = 0.8,
    fuse_labels: list[str] | None = None,
    run: str | None = None,
    project: str = cast("str", CFG.active_project),
    overwrite: bool = False,
    verbose: bool = False,
):
    """Segment images with MaskFormer.

    Args:
        image_path: Path to the images to be segmented. If not provided uses all downloaded images in the project.
        labels: Labels to focus on.
        batch_size: Batch size for the segmentation model.
        model_id: Mask2Former model to load.
        threshold: The probability score threshold to keep predicted instance masks.
        mask_threshold: Threshold to use when turning the predicted masks into binary values.
        overlap_threshold: The overlap mask area threshold to merge or discard small
            disconnected parts within each binary instance mask.
        fuse_labels: The labels in this state will have all their instances fused together.
        run: Model run ID.
        project: The project to use. Uses the active project by default.
        overwrite: Overwrite an existing run.
        verbose: Print verbose log to the terminal. Useful for debugging models.
    """
    # Open the project
    proj = Project(project)

    # Save the run metadata.
    # ==================================================
    model = "maskformer"
    model_params = {
        "model_id": model_id,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "overlap_mask_area_threshold": overlap_threshold,
        "labels_to_fuse": fuse_labels,
    }

    result = proj.add_run(run, model, model_params, overwrite)
    run = result.get("run")[0]

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
    processed, unprocessed = proj.get_segmentation_status(uids, run)

    if len(unprocessed) == 0:
        logger.info(f"Nothing to process.")
        return

    if labels is None:
        labels = list(MaskFormer.id_to_label.values())

    handle = serve_model(model, verbose, **model_params)
    logger.info(f"Segmenting {len(unprocessed)} images using {model}...")
    batches = list(batched(unprocessed, batch_size))
    for batch_idx, batch in enumerate(batches, 1):

        # Extract the paths and open the images as NumPy arrays.
        request = {
            "images": [],
            "labels": labels,
        }
        for uid in batch:
            path, source = unprocessed[uid]
            img_data = {
                "uid": uid,
                "image": oj.dumps(
                    np.asarray(iio.imread(path)), option=oj.OPT_SERIALIZE_NUMPY
                ),
            }
            request["images"].append(img_data)

        # Process the images.
        logger.info(f"Segmenting batch [{batch_idx:>4d}/{len(batches):>4d}]...")
        responses = handle.remote(request).result()
        logger.info(f"Successfully segmented {len(batch)} images, saving instances.")

        # Save the instances.
        segmentations = []
        for response in responses:
            instances = oj.loads(response.instances)
            segmentations.append(
                {
                    "run": run,
                    "image": response.uid,
                    "labels": response.labels,
                    "polygons": mask2poly(instances, model="maskformer"),
                }
            )
        # Update the segmentation table.
        proj.add_segmentations(segmentations, overwrite)
