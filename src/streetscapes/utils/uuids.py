"""Hash- and UUID-related utilities."""

import uuid
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid7 as __uuid7

import filetype as ft
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def uuid7(as_str: bool = False) -> uuid.UUID | str:
    """Return a UUID7 instance, optionally converted to string.

    Args:
        as_str: If True, convert the UUID to string before returning.

    Returns:
        The UUID.
    """
    u = __uuid7()
    return u if not as_str else str(u)


def get_image_hash(image: str | Path | bytes) -> bytes:
    """Get the SHA-256 hash of an image file.

    Args:
        image: The path to the file or raw bytes.

    Returns:
        SHA-256 digest.
    """
    if not ft.is_image(image):
        raise ValueError("The provided file is not an image.")

    if isinstance(image, bytes):
        import io

        image = io.BytesIO(image)  # type: ignore[assignment]

    return sha256(np.asarray(Image.open(image))).digest()


def hash2uuid(ihash: bytes) -> uuid.UUID:
    """Create a UUID (128 bits) from a SHA-256 hash of an image file.

    Args:
        ihash: The hash.

    Returns:
        A UUID.
    """
    return uuid.UUID(ihash.hex()[::2])


def get_image_uuid(image: str | Path | bytes) -> uuid.UUID:
    """Get the unique and reproducible UUID of an image file.

    Args:
        image: The path to the file or raw bytes.

    Returns:
        Image UUID.
    """
    if not ft.is_image(image):
        msg = "Input image type of is not supported!"
        raise ValueError(msg)

    return hash2uuid(get_image_hash(image))
