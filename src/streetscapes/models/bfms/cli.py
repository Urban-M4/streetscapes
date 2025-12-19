"""Command line interface for BFMS model."""

from pathlib import Path

import filetype as ft

from streetscapes import config
from streetscapes.models.bfms.db import SCHEMA
from streetscapes.models.bfms.functions import segment_images
from streetscapes.project import Project


def cli(
    image_path: str,  # TODO: take names subset rather than or in addition to directory / path
    overwrite: bool = False,
):
    """Segment images with the BFMS model."""
    model_params = {}  # TODO: figure out if we can tweak any params

    if image_path is not None:
        image_path = Path(image_path)

    if image_path.is_dir():
        image_path = [
            im_path for im_path in image_path.glob("*.*") if ft.is_image(im_path)
        ]

    project = Project(config.get("active_project"))

    project.ensure_table("bfms", SCHEMA)

    (processed, unprocessed) = project.get_image_status(image_path, "bfms", overwrite)

    response = segment_images(unprocessed, model_params)

    # TODO:
    # Store the segmentations and their metadata
    # save_segmentations(project, model_params, response, processed)


from itertools import batched
from uuid import UUID
import orjson as oj

from streetscapes.serve.server import serve_model


def segment_images(
    images: list[tuple[bytes, UUID]],
    batch_size: int = 10,
    model_params: dict = None,
):
    """Segment a collection of images.

    Args:
        image_paths: A list of paths to individual images.
        batch_size: Batch size to use.
        model_params: Model parameters to pass to the model instance.
        overwrite: Overwrite processed images.
        project: The project to open.

    """
    if model_params is None:
        model_params = {}

    handle = serve_model("bfms", **model_params)

    for entries in batched(images, batch_size):
        hashes = [e[0] for e in entries]
        images = [np.asarray(iio.imread(e[1])) for e in entries]

        data = {
            "images": [
                {
                    "hash": h,
                    "image": oj.dumps(image, option=oj.OPT_SERIALIZE_NUMPY),
                }
                for h, image in zip(hashes, images)
            ],
            "labels": labels,
        }
        response = handle.remote(data).result()

        return response
