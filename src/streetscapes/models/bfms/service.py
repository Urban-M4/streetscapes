"""BFMS segmentation service."""

import imageio.v3 as iio
import numpy as np
import shapely
from pydantic import BaseModel

from streetscapes.models.bfms.model import BFMS
from streetscapes.utils.masks import mask2poly


class BFMSRequest(BaseModel):
    image: bytes  # encoded image file (e.g. JPEG)


class BFMSResponse(BaseModel):
    labels: list[str]  # Instance labels
    polygons: bytes  # WKB-encoded shapely GeometryCollection


class BFMSService:
    """Inference service for the BFMS model.

    Exposes BFMS inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(self, model_id: str):
        """Inference service for the BFMS model.

        model_id: Huggingface model ID.
        """
        self.model = BFMS(model_id=model_id)

    def handle(self, request: dict) -> BFMSResponse:
        """Run a segmentation request."""
        req = BFMSRequest(**request)

        image = np.asarray(iio.imread(req.image))
        result = self.model.segment(image)

        # Convert masks to polygons here to avoid (de)serializing the masks.
        polygons = mask2poly(result["instances"], model="bfms", image=image)

        response = BFMSResponse(
            labels=result["labels"],
            polygons=shapely.to_wkb(polygons),
        )

        return response
