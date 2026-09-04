"""SAM3 segmentation service."""

import uuid

from pydantic import BaseModel
from ray import cloudpickle

from streetscapes.models.sam3.model import SAM3


class SAM3Image(BaseModel):
    uid: uuid.UUID
    image: bytes


class SAM3Request(BaseModel):
    images: list[SAM3Image]
    prompt: str | list[str]


class SAM3Response(BaseModel):
    uid: uuid.UUID
    labels: list[str]
    confidences: list[float]
    instances: bytes


class SAM3Service:
    """Inference service for the SAM3 model.

    Exposes SAM3 inferece as a structured request/response
    interface usable by Ray Serve.

    NOTE: The weights for SAM3 need to be downloaded manually!
    """

    def __init__(
        self,
        weights: str = "sam3.pt",
        device: str | None = None,
        confidence: float = 0.25,
        quantisation: str | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the SAM3 segmentation service."""
        self.model = SAM3(weights, device, confidence, quantisation, *args, **kwargs)

    def handle(self, request: dict) -> list[SAM3Response]:
        """Handle segmentation request."""
        req = SAM3Request(**request)

        uids = []
        images = []
        for entry in req.images:
            uids.append(entry.uid)
            images.append(cloudpickle.loads(entry.image))

        # Segment the images
        segmentations = self.model.segment_images(uids, images, req.prompt)

        # Construct the response
        response = []
        for result in segmentations:
            result["instances"] = cloudpickle.dumps(result["instances"])
            response.append(SAM3Response(**result))

        return response
