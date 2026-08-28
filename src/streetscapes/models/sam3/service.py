import numpy as np
import orjson as oj
from pydantic import BaseModel
import uuid
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
        self.model = SAM3(weights, device, confidence, quantisation, *args, **kwargs)

    def handle(self, request: dict) -> list[SAM3Response]:
        req = SAM3Request(**request)

        uids = []
        images = []
        for entry in req.images:
            uids.append(entry.uid)
            images.append(np.array(oj.loads(entry.image)))

        # Segment the images
        segmentations = self.model.segment_images(uids, images, req.prompt)

        # Construct the response
        response = []
        for result in segmentations:
            result["instances"] = oj.dumps(
                result["instances"], option=oj.OPT_SERIALIZE_NUMPY
            )
            response.append(SAM3Response(**result))

        return response
