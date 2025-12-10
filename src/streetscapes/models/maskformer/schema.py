from pydantic import BaseModel
from pathlib import Path
import numpy as np


class MaskFormerImageSchema(BaseModel):
    hash: bytes
    image: bytes


class MaskFormerRequestSchema(BaseModel):
    images: list[MaskFormerImageSchema]
    labels: dict


class MaskFormerResponseSchema(BaseModel):
    image_hash: bytes
    instances: dict[int, str]
    masks: dict[int, list]
