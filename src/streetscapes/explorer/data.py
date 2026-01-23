from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Bbox(BaseModel):
    n: float = Field(default=90, ge=-90, le=90)
    e: float = Field(default=180, ge=-180, le=180)
    s: float = Field(default=-90, ge=-90, le=90)
    w: float = Field(default=-180, ge=-180, le=180)


class FilterParams(Bbox):
    ratings: list[int] = Field(default=[])
    model_runs: list[str] = Field(default=[])
    sources: list[str] = Field(default=[])
    tags: list[str] = Field(default=[])
    compass_angle: list[float] = Field(default=[0, 360])
    date_range: tuple[datetime, datetime] = Field(default=(datetime(1826,1,1),datetime.now()))
    panoramic: list[int] = Field(default=[])


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
