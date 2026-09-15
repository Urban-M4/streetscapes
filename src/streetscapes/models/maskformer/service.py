"""Maskformer inference service."""

import uuid

import shapely
from pydantic import BaseModel
from ray import cloudpickle

from streetscapes.models.maskformer.model import MaskFormer
from streetscapes.utils.masks import mask2poly


class MaskFormerImage(BaseModel):
    uid: uuid.UUID
    image: bytes


class MaskFormerRequest(BaseModel):
    images: list[MaskFormerImage]


class MaskFormerResponse(BaseModel):
    uid: uuid.UUID
    labels: list[str]
    confidences: list[float]
    polygons: bytes  # WKB-encoded shapely GeometryCollection


class MaskFormerService:
    """Inference service for the MaskFormer model.

    Exposes MaskFormer inferece as a structured request/response
    interface usable by Ray Serve.
    """

    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        labels_to_fuse: list[str | int] | None = None,
        device: str | None = None,
    ):
        """TODO: add docstring."""
        self.model = MaskFormer(
            model_id,
            threshold,
            mask_threshold,
            overlap_mask_area_threshold,
            labels_to_fuse,
            device,
        )

    def handle(self, request: dict) -> list[MaskFormerResponse]:
        """TODO: add docstring."""
        # Convert the request into a schema to validate it.
        schema = MaskFormerRequest(**request)

        uids = []
        images = []
        for entry in schema.images:
            uids.append(entry.uid)
            images.append(cloudpickle.loads(entry.image))

        # Segment the images
        segmentations = self.model.segment_images(
            uids,
            images,
        )

        # Construct the response
        response = []
        for result in segmentations:
            # Convert masks to polygons here to avoid (de)serializing the masks.
            polygons = mask2poly(result.pop("instances"), model="maskformer")
            result["polygons"] = shapely.to_wkb(polygons)
            response.append(MaskFormerResponse(**result))

        return response
