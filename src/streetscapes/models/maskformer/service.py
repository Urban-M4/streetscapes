import numpy as np
import orjson as oj
from pydantic import BaseModel

from streetscapes.models.maskformer.model import MaskFormer


class MaskFormerImageSchema(BaseModel):
    hash: bytes
    image: bytes


class MaskFormerRequestSchema(BaseModel):
    images: list[MaskFormerImageSchema]
    labels: dict


class MaskFormerResponseSchema(BaseModel):
    image_hash: bytes
    labels: list[str]
    instances: bytes


class MaskFormerService:
    """Inference service for the MaskFormer model.

    Exposes MaskFormer inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(self):
        self.model = MaskFormer()

    def handle(self, request: dict) -> dict:
        # Convert the request into a schema to validate it.
        schema = MaskFormerRequestSchema(**request)

        hashes = []
        images = []
        for entry in schema.images:
            hashes.append(entry.hash)
            images.append(np.array(oj.loads(entry.image)))

        # Segment the images
        segmentations = self.segment_images(
            hashes,
            images,
            schema.labels,
        )

        # Construct the response schemata
        response = []
        for result in segmentations:
            result["instances"] = oj.dumps(
                result["instances"], option=oj.OPT_SERIALIZE_NUMPY
            )
            response.append(MaskFormerResponseSchema(**result))

        return response
