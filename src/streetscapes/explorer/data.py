from dataclasses import dataclass, field
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
    id: str
    url: str
    lat: float
    lon: float


@dataclass
class Instance:
    label: str
    polygon: list[list[tuple[float, float]]] = field(default_factory=list)


@dataclass
class Segmentation:
    model_name: str
    id: str  # archive
    run_args: str
    instances: list[Instance] = field(default_factory=list)
    notes: str = ""


@dataclass
class ImageMetadata(Image):
    width: int
    height: int
    altitude: Optional[float] = None
    captured_at: Optional[datetime] = None
    panoramic: Optional[int] = None
    source: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    rating: Optional[int] = None
    compass_angle: Optional[float] = None
    notes: str = ""
    segmentation: list[Segmentation] = field(default_factory=list)

@dataclass
class AggregateStats:
    tags: list[str]
    labels: list[str]  # unique labels in any segmentation
    model_run_names: list[str]
    image_sources: list[str]
    date_range: tuple[datetime, datetime]
    models: list[str]
