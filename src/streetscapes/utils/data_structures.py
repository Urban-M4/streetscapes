"""Segmentation data structures."""

from dataclasses import dataclass, field
from typing import TypeVar

from shapely import Polygon

T = TypeVar("T", tuple[float, float], Polygon)


@dataclass
class Instance[T]:
    """Segmentation instance."""

    label: str
    polygon: list[T] = field(default_factory=list)


@dataclass
class Segmentation[T]:
    """Segmentation of an image."""

    model_name: str
    id: str  # archive
    run_args: str
    rating: int
    instances: list[Instance[T]] = field(default_factory=list)
    notes: str = ""
