"""SAM3 segmentation service."""

import uuid

import imageio.v3 as iio
import numpy as np
import shapely
from pydantic import BaseModel

from streetscapes.models.sam3.model import SAM3
from streetscapes.utils.masks import mask2poly


class SAM3Image(BaseModel):
    uid: uuid.UUID
    image: bytes  # encoded image file (e.g. JPEG)


class SAM3Request(BaseModel):
    images: list[SAM3Image]
    prompt: str | list[str]


class SAM3Response(BaseModel):
    uid: uuid.UUID
    labels: list[str]
    confidences: list[float]
    polygons: bytes  # WKB-encoded shapely GeometryCollection


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
            images.append(np.asarray(iio.imread(entry.image)))

        # Segment the images
        segmentations = self.model.segment_images(uids, images, req.prompt)

        # Construct the response
        response = []
        for result in segmentations:
            # Convert masks to polygons here to avoid (de)serializing the masks.
            polygons = mask2poly(result.pop("instances"), model="dinosam")
            result["polygons"] = shapely.to_wkb(polygons)
            response.append(SAM3Response(**result))

        return response
