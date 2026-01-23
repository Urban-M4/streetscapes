"""FastAPI server implementation."""

from datetime import datetime
from typing import Annotated, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Query

from streetscapes import config
from streetscapes.explorer.data import (
    AggregateStats,
    Bbox,
    FilterParams,
    Image,
    ImageMetadata,
)
from streetscapes.explorer.dummy_data import _images

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://urban-m4.github.io/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _inbounds(img: Image | ImageMetadata, bbox: Bbox) -> bool:
    # Temporary implementation.
    # Full implementation needs to account for spherical coordinates properly
    if bbox.n > img.lat > bbox.s and bbox.w > img.lon > bbox.e:
        return True
    return False


def _fetch_images(bbox: Optional[Bbox]) -> list[Image]:
    if bbox is not None:
        return [Image(img) for img in _images if _inbounds(img, bbox)]
    return _images


def _unknown_image(image_id):
    msg = f"No image found with id '{image_id}'"
    raise HTTPException(status_code=404, detail=msg)


@app.get("/")
async def root():
    """Server root."""
    return {"message": "Welcome to the Streetscapes Explorer"}


@app.get("/project")
async def project():
    """Get the active project name."""
    return config.get("active_project")


@app.get("/stats")
async def fetch_stats() -> AggregateStats:
    """Get the aggregate stats of the images."""
    return AggregateStats(
        tags=["birb",],
        labels=["tree", "car", "building", "bike", "person"],
        model_run_names=["manual"],
        image_sources=[
            "wikimedia-commons",
            "mappilary",
        ],
        date_range=(datetime(2026, 1, 19), datetime(2026, 1, 20)),
        models=["DinoSAM", "maskformer", "bfms", "manual"]
    )


@app.get("/images")
async def fetch_images(filter: Annotated[FilterParams, Query()]
) -> list[Image]:
    """Fetch streetscape images corresponding to a bounding box and optionally filters."""
    # bbox = Bbox(**filter.model_dump())
    return _fetch_images(None)


@app.get("/images/{image_id}")
async def fetch_image_metadata(image_id: str) -> ImageMetadata :
    """Get all metadata associated with a certain image, including segmentations."""
    for img in _images:
        if img.id == image_id:
            return img
    _unknown_image(image_id)


@app.post("/images/{image_id}/rating")
async def set_rating(image_id: str, rating: int | None):
    """Set an image's rating."""
    for img in _images:
        if img.id == image_id:
            img.rating = rating
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/tags")
async def set_tags(image_id: str, tags: list[str]):
    """Set an image's tags."""
    for img in _images:
        if img.id == image_id:
            img.tags = tags
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/notes")
async def set_notes(image_id: str, notes: str):
    """Set an image's notes."""
    for img in _images:
        if img.id == image_id:
            img.notes = notes
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/{segmentation_id}/{instance_idx}/{label}")
async def set_instance_label(
    image_id: str, segmentation_id: str, instance_idx: int, label: str
):
    """Set the label of a specific instance within a segmentation."""
    pass

@app.post("/images/{image_id}/segment/{model}/{run_args}")
async def segment_image(image_id, model, run_args):
    """Compute a new segmentation of an image."""
    pass


def serve():
    """Start the Streetscapes Explorer server."""
    print("Go to https://urban-m4.github.io/Urban-M5/ and paste in https://0.0.0.0:5000 as web service.")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
    # TODO: automatically launch browser. docs.python.org/3/library/webbrowser.html
