"""Helpers shared by the image source clients."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import shapely

from streetscapes import utils
from streetscapes.utils.sun import solar_altitude

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from datetime import datetime

    import requests

    from streetscapes.utils.metadata import ImageMeta

logger = logging.getLogger(__name__)

# How high the sun has to be, in degrees, for an image to count as captured in
# daylight. Any lower, and its colours are those of dusk rather than of the scene.
MIN_SUN_ALTITUDE = 2.0


class CapturedImage(Protocol):
    """An image record that says when and where it was captured."""

    captured_at: datetime | None
    geometry: str


_Image = TypeVar("_Image", bound=CapturedImage)


def captured_in_daylight(images: Sequence[CapturedImage]) -> list[bool]:
    """Tell which images were captured with the sun at least `MIN_SUN_ALTITUDE` high.

    An image without a capture time cannot be placed, so it does not count as
    captured in daylight.

    Args:
        images: Validated image records, with a WKT point as their geometry.

    Returns:
        A boolean mask over the images.
    """
    points = shapely.from_wkt([image.geometry for image in images])
    # A missing capture time gives a NaN altitude, which compares as False.
    return [
        solar_altitude(image.captured_at, point.x, point.y) >= MIN_SUN_ALTITUDE
        for image, point in zip(images, points, strict=True)
    ]


def keep_daytime(images: list[_Image]) -> list[_Image]:
    """Keep the images captured with the sun at least `MIN_SUN_ALTITUDE` high.

    Args:
        images: Validated image records, with a WKT point as their geometry.

    Returns:
        The images captured in daylight, see `captured_in_daylight`.
    """
    mask = captured_in_daylight(images)
    kept = [image for image, daytime in zip(images, mask, strict=True) if daytime]

    logger.debug(f"Dropped {len(images) - len(kept)} images not captured in daylight.")
    return kept


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


class RateLimiter:
    """Hands out permission to make a request, at most so many per minute.

    Requests are spaced out evenly rather than let through in bursts, so that a
    quota stays intact however many threads are asking for it at once.
    """

    def __init__(self, per_minute: int):
        """Instantiate the limiter.

        Args:
            per_minute: How many requests are allowed per minute. A limit of 0
                (or less) lets every request through immediately.
        """
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        """Wait until the next request is allowed to be made."""
        if not self.interval:
            return

        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval

        if delay > 0:
            time.sleep(delay)


_In = TypeVar("_In")
_Out = TypeVar("_Out")


def concurrent_map(
    func: Callable[[_In], _Out],
    items: Iterable[_In],
    workers: int,
) -> Iterator[_Out]:
    """Apply a function to every item in a pool of threads.

    Meant for work that waits on the network, where threads help despite the
    GIL. Only a limited number of items are in flight at any one time, so
    `items` may be an arbitrarily long (or lazy) iterable.

    Args:
        func: The function to apply. It is called from several threads at once,
            so it has to be safe to call concurrently.
        items: The items to apply it to, consumed lazily.
        workers: How many threads to use. One means no threads at all, which
            keeps the work (and any traceback) on the calling thread.

    Yields:
        The results, in the order in which they are finished rather than in the
        order of the items.
    """
    if workers <= 1:
        yield from map(func, items)
        return

    # Keeping the pool fed takes a couple of items per thread; more than that
    # would only queue up work that cannot be started any sooner.
    window = 2 * workers
    items = iter(items)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(func, item) for item in islice(items, window)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Top the pool up before handing the results over, so that the
            # threads keep working while the caller deals with them.
            pending.update(pool.submit(func, item) for item in islice(items, len(done)))
            for future in done:
                yield future.result()
