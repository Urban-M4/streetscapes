"""DinoSAM model inference service."""

import uuid

import imageio.v3 as iio
import numpy as np
import shapely
from pydantic import BaseModel

from streetscapes.models.dinosam.model import DinoSAM
from streetscapes.utils.masks import mask2poly


class DinoSAMImage(BaseModel):
    uid: uuid.UUID
    image: bytes  # encoded image file (e.g. JPEG)


class DinoSAMRequest(BaseModel):
    images: list[DinoSAMImage]
    prompt: str | list[str]


class DinoSAMResponse(BaseModel):
    uid: uuid.UUID
    labels: list[str]
    confidences: list[float]
    polygons: bytes  # WKB-encoded shapely GeometryCollection


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
            images.append(np.asarray(iio.imread(entry.image)))

        # Segment the images
        segmentations = self.model.segment_images(uids, images, req.prompt)

        # Construct the response
        response = []
        for result in segmentations:
            # Convert masks to polygons here to avoid (de)serializing the masks.
            polygons = mask2poly(result.pop("instances"), model="dinosam")
            result["polygons"] = shapely.to_wkb(polygons)
            response.append(DinoSAMResponse(**result))

        return response
