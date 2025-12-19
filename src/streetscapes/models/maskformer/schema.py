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
    labels: list[str]
    instances: bytes
