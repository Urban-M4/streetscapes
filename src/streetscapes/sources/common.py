"""Helpers shared by the image source clients."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from streetscapes import utils

if TYPE_CHECKING:
    import uuid

    import requests

    from streetscapes.utils.metadata import ImageMeta


def download_image(
    session: requests.Session,
    url: str,
    output_dir: str | Path | None,
    image_id: int | str | None,
    source: str,
    uid: uuid.UUID | None = None,
    skip_existing: bool = True,
) -> ImageMeta:
    """Download an image from a URL into a source's image directory.

    Images are named after the UUID derived from their content, so a downloaded
    image can only be recognised again via the `<image_id>` file holding that
    UUID, which is written alongside it.

    Args:
        session: The (authenticated) session to download with.
        url: The download URL.
        output_dir: Destination directory.
        image_id: The image ID at the source.
        source: The source name (e.g. 'mapillary', 'kartaview').
        uid: Image UUID (from the SHA-256 hash), if it is already known.
        skip_existing: Don't re-download existing images.

    Returns:
        Image metadata.
    """
    output_path = output_dir
    if output_dir is not None:
        output_dir = Path(output_dir)

    content = None
    if uid is not None:
        if output_dir is not None:
            image_path = list(output_dir.glob(f"*{uid}*"))
            if len(image_path) > 0:
                content = image_path[0].read_bytes()
        if content is None:
            # The image is missing, download it again.
            skip_existing = False

    if uid is None or not skip_existing:
        response = session.get(url)
        response.raise_for_status()
        content = response.content

    if content is None:
        raise ValueError(f"Failed to download image with UUID '{uid}': empty content")

    meta = utils.get_image_metadata(content)

    if uid is None and output_dir is not None:
        utils.ensure_dir(output_dir)
        output_path = output_dir / f"{meta.uid}.{meta.ext}"
        output_path.write_bytes(meta.content)
        # write source-id -> uuid mapping
        if image_id is not None:
            with (output_dir / str(image_id)).open("w") as f:
                f.write(str(meta.uid))

    meta.fpath = output_path
    meta.source = source

    return meta
