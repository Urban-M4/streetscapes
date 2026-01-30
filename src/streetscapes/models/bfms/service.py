import numpy as np
import orjson as oj
from pydantic import BaseModel
import uuid
from streetscapes.models.bfms.model import BFMS


class BFMSRequest(BaseModel):
    image: str  # JSON-encoded numpy array


class BFMSResponse(BaseModel):
    uid: uuid.UUID | None = None
    mask: str  # JSON-encoded numpy array


class BFMSService:
    """Inference service for the BFMS model.

    Exposes BFMS inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(self):
        self.model = BFMS()

    def handle(self, request: dict) -> dict:
        req = BFMSRequest(**request)

        image = np.array(oj.loads(req.image), dtype=np.uint8)

        mask = self.model.segment(image)

        response = BFMSResponse(
            mask=oj.dumps(
                mask,
                option=oj.OPT_SERIALIZE_NUMPY,
            )
        )

        return response
