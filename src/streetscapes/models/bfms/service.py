import numpy as np
import orjson as oj
from pydantic import BaseModel

from streetscapes.models.bfms.model import BFMS


class BFMSRequest(BaseModel):
    image: str  # JSON-encoded numpy array


class BFMSResponse(BaseModel):
    labels: list[str]  # Instance labels
    instances: bytes  # JSON-encoded numpy array


class BFMSService:
    """Inference service for the BFMS model.

    Exposes BFMS inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(self, model_id: str):
        self.model = BFMS(model_id=model_id)

    def handle(self, request: dict) -> BFMSResponse:
        req = BFMSRequest(**request)

        image = np.array(oj.loads(req.image), dtype=np.uint8)
        result = self.model.segment(image)

        response = BFMSResponse(
            labels=result["labels"],
            instances=oj.dumps(
                result["instances"],
                option=oj.OPT_SERIALIZE_NUMPY,
            ),
        )

        return response
