"""BFMS segmentation service."""

from pydantic import BaseModel
from ray import cloudpickle

from streetscapes.models.bfms.model import BFMS


class BFMSRequest(BaseModel):
    image: bytes  # cloudpickled numpy array


class BFMSResponse(BaseModel):
    labels: list[str]  # Instance labels
    instances: bytes  # cloudpickled numpy array


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

        image = cloudpickle.loads(req.image)
        result = self.model.segment(image)

        response = BFMSResponse(
            labels=result["labels"],
            instances=cloudpickle.dumps(result["instances"]),
        )

        return response
