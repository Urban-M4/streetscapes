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
