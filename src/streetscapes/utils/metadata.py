"""Metadata."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from pathlib import Path


@dataclass
class ImageMeta:
    """Object holds all relevant info to identify/find a streetscapes image."""

    content: bytes
    hash: bytes
    uid: uuid.UUID
    ext: str
    fpath: str | Path | None = None
    source: str | None = None
    shard: str | Path | None = None
