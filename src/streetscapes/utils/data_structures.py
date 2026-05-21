from dataclasses import dataclass, field


@dataclass
class Instance:
    label: str
    polygon: list[list[tuple[float, float]]] = field(default_factory=list)


@dataclass
class Segmentation:
    model_name: str
    id: str  # archive
    run_args: str
    rating: int
    instances: list[Instance] = field(default_factory=list)
    notes: str = ""
