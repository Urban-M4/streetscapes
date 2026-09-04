"""Metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import filetype as ft

from streetscapes.utils.uuids import get_image_hash, hash2uuid

if TYPE_CHECKING:
    import uuid


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


def get_image_metadata(image: bytes | str | Path) -> ImageMeta:
    """Get some reproducible image metadata.

    Args:
        image: Binary content or a path to an existing image.

    Returns:
        An object contiaining the image metadata.
    """
    _hash = get_image_hash(image)
    _uuid = hash2uuid(_hash)
    ext = ft.guess_extension(image).lower()

    if isinstance(image, (str, Path)):
        image = Path(image).read_bytes()

    return ImageMeta(image, _hash, _uuid, ext)
