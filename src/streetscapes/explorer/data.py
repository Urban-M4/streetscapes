from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated

from annotated_types import Ge, Le
from pydantic import BaseModel, Field

from streetscapes.utils.data_structures import Segmentation


class Bbox(BaseModel):
    n: float = Field(default=90, ge=-90, le=90)
    e: float = Field(default=180, ge=-180, le=180)
    s: float = Field(default=-90, ge=-90, le=90)
    w: float = Field(default=-180, ge=-180, le=180)


class FilterParams(Bbox):
    # image level filters
    image_ratings: list[int] = []
    sources: list[str] = []
    tags: list[str] = []
    # metadata level filters
    date_range: tuple[datetime, datetime] = Field(default=(datetime(1826,1,1),datetime.now()))
    # segmentation level filters
    models: list[str] = []
    model_runs: list[str] =[]
    labels: list[str] = []
    segmentation_ratings: list[Annotated[int, Ge(0), Le(5)]] = Field(default=[])


@dataclass
class Image:
    id: str
    lat: float
    lon: float


@dataclass
class ImageMetadata(Image):
    width: int
    height: int
    altitude: float | None = None
    captured_at: datetime | None = None
    panoramic: int | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    rating: int | None = None
    compass_angle: float | None = None
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
