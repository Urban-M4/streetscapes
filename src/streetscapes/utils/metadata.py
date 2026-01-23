from dataclasses import dataclass
import io
import uuid
from pathlib import Path

@dataclass
class ImageMeta:

    content: bytes
    ihash: bytes
    iuuid: uuid.UUID
    ext: str
    source: str | None = None
    path: str | Path | None = None
    shard: str | None = None
