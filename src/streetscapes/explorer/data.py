from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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
    polygon: list[list[tuple[float, float]]]


@dataclass
class Segmentation:
    model_name: str
    id: str  # archive
    run_args: str
    instances: tuple[Instance]
    notes: str = ""


@dataclass
class ImageMetadata(Image):
    width: int
    height: int
    altitude: Optional[float] = None
    captured_at: Optional[datetime] = None
    panoramic: Optional[int] = None
    source: Optional[str] = None
    tags: tuple[str] = ()
    rating: Optional[int] = None
    compass_angle: Optional[float] = None
    notes: str = ""
    segmentation: tuple[Segmentation] = ()


@dataclass
class AggregateStats:
    tags: list[str]
    labels: list[str]  # unique labels in any segmentation
    model_run_names: list[str]
    image_sources: list[str]
    date_range: tuple[datetime, datetime]
