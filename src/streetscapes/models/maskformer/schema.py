from pydantic import BaseModel
from pathlib import Path


class MaskFormerRequestSchema(BaseModel):
    image_path: str | Path
    labels: dict
    batch_size: int = 10


class MaskFormerResponseSchema(BaseModel):
    image_path: str | Path
    instances: dict[int, str]
    masks: dict[int, list]
