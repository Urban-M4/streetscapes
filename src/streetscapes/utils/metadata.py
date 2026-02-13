from dataclasses import dataclass
import uuid
from pathlib import Path

@dataclass
class ImageMeta:

    content: bytes
    hash: bytes
    uid: uuid.UUID
    ext: str
    fpath: str | Path | None = None
    source: str | None = None
    shard: str | Path | None = None
