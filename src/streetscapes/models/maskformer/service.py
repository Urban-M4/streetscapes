import numpy as np
import orjson as oj
from pydantic import BaseModel
import uuid
from streetscapes.utils import logger
from streetscapes.models.maskformer.model import MaskFormer


class MaskFormerImage(BaseModel):
    uid: uuid.UUID
    image: bytes


class MaskFormerRequest(BaseModel):
    images: list[MaskFormerImage]
    labels: list[str]


class MaskFormerResponse(BaseModel):
    uid: uuid.UUID
    labels: list[str]
    instances: bytes


class MaskFormerService:
    """Inference service for the MaskFormer model.

    Exposes MaskFormer inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        labels_to_fuse: list[str | int] | None = None,
        device: str | None = None,
    ):
        self.model = MaskFormer(
            model_id,
            threshold,
            mask_threshold,
            overlap_mask_area_threshold,
            labels_to_fuse,
            device,
        )

    def handle(self, request: dict) -> list[MaskFormerResponse]:
        # Convert the request into a schema to validate it.
        schema = MaskFormerRequest(**request)

        uids = []
        images = []
        for entry in schema.images:
            uids.append(entry.uid)
            images.append(np.array(oj.loads(entry.image)))

        # Segment the images
        segmentations = self.model.segment_images(
            uids,
            images,
            schema.labels,
        )

        # Construct the response
        response = []
        for result in segmentations:
            result["instances"] = oj.dumps(
                result["instances"], option=oj.OPT_SERIALIZE_NUMPY
            )
            response.append(MaskFormerResponse(**result))

        return response
