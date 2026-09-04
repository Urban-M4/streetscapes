"""DinoSAM model inference service."""

import uuid

from pydantic import BaseModel
from ray import cloudpickle

from streetscapes.models.dinosam.model import DinoSAM


class DinoSAMImage(BaseModel):
    uid: uuid.UUID
    image: bytes


class DinoSAMRequest(BaseModel):
    images: list[DinoSAMImage]
    prompt: str | list[str]


class DinoSAMResponse(BaseModel):
    uid: uuid.UUID
    labels: list[str]
    confidences: list[float]
    instances: bytes


class DinoSAMService:
    """Inference service for the DinoSAM model.

    Exposes DinoSAM inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(
        self,
        sam_model_id: str = "facebook/sam2.1-hiera-large",
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        *args,
        **kwargs,
    ):
        """Initialize DinoSAM service."""
        self.model = DinoSAM(
            sam_model_id, dino_model_id, box_threshold, text_threshold, *args, **kwargs
        )

    def handle(self, request: dict) -> list[DinoSAMResponse]:
        """Handle a segmentation request to DinoSAM."""
        req = DinoSAMRequest(**request)

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
            response.append(DinoSAMResponse(**result))

        return response
