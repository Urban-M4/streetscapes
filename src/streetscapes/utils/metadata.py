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
    fpath: str | Path | None = None
    source: str | None = None
    shard: str | Path | None = None
