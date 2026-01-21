"""FastAPI server implementation."""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@dataclass
class Bbox:
    n: float
    e: float
    s: float
    w: float


@dataclass
class Image:
    id: int
    url: str
    lat: float
    lon: float


@dataclass
class Instance:
    label: str
    polygon: str  # json multi-polygon string


@dataclass
class Segmentation:
    model_name: str
    run_args: str
    instances: tuple[Instance]
    notes: str = ""


@dataclass
class ImageMetadata(Image):
    width: int
    height: int
    altitude: Optional[float] = None
    creation_date: Optional[datetime] = None
    panoramic: Optional[int] = None
    source: Optional[str] = None
    tags: tuple[str] = ()
    rating: Optional[int] = None
    compass_angle: Optional[float] = None
    notes: str = ""
    segmentation: tuple[Segmentation] = ()


_images = [
    ImageMetadata(
        id=0,
        url=r"https://upload.wikimedia.org/wikipedia/commons/0/00/Zaden_van_een_Gele_lis_%28Iris_pseudacorus%29._06-03-2024._%28d.j.b.%29.jpg",
        lat=52.3751914,
        lon=4.8954506,
        altitude=0.0,
        creation_date=datetime(2026, 1, 20),
        source="wikimedia-commons",
        width=3454,
        height=5182,
        notes="Gele Iris",
    ),
    ImageMetadata(
        id=1,
        url=r"https://upload.wikimedia.org/wikipedia/commons/b/b8/Chestnut-naped_antpitta_%28Grallaria_nuchalis_ruficeps%29_Las_Tangaras.jpg",
        lat=52.3727217,
        lon=4.9003963,
        altitude=0.0,
        creation_date=datetime(2026, 1, 19),
        source="wikimedia-commons",
        tags=["birb",],
        width=3092,
        height=4000,
        notes="is cute",
    ),
]


def _inbounds(img: Image | ImageMetadata, bbox: Bbox) -> bool:
    # Temporary implementation.
    # Full implementation needs to account for spherical coordinates properly
    if bbox.n > img.lat > bbox.s and bbox.w > img.lon > bbox.e:
        return True
    return False


@dataclass
class AggregateStats:
    tags: list[str]
    model_run_names: list[str]
    image_sources: list[str]
    date_range: tuple[datetime, datetime]


def _fetch_images(bbox: Bbox) -> list[Image]:
    return [Image(img) for img in _images if _inbounds(img, bbox)]


def _unknown_image(image_id):
    msg = f"No image found with id '{image_id}'"
    raise ValueError(msg)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/project")
async def projects():
    return "placeholder-project-name"


@app.post("/stats")
async def fetch_stats(bbox: Bbox) -> AggregateStats:
    return AggregateStats(
        ["birb"], [], ["wikimedia-commons",], (datetime(2026, 1, 19), datetime(2026, 1, 20))
    )


@app.post("/images")
async def fetch_images(
    bbox: Bbox,
    filters: Optional[dict[str, Any]] = None
) -> list[Image]:
    """Fetch streetscape images corresponding to a bounding box and optionally filters."""
    return _fetch_images(bbox)


@app.post("/images/{image_id}")
async def fetch_image_metadata(image_id: int):
    for img in _images:
        if img.id == image_id:
            return img
    _unknown_image(image_id)


@app.post("/images/{image_id}/rating")
async def set_rating(image_id: int, rating: int | None):
    for img in _images:
        if img.id == image_id:
            img.rating = rating
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/tags")
async def set_tags(image_id: int, tags: tuple[str]):
    for img in _images:
        if img.id == image_id:
            img.tags = tags
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/notes")
async def set_notes(image_id: int, notes: str):
    for img in _images:
        if img.id == image_id:
            img.notes = notes
            return None
    _unknown_image(image_id)


def serve():
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
