"""Segmentation mask utils."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from shapely import GeometryCollection, geometry
from skimage.measure import find_contours

if TYPE_CHECKING:
    from collections.abc import Iterator


def _norm_contours(contours: list[np.ndarray]):
    """Normalize contours to remove effect of padding."""
    return [np.maximum(contour - 1, 0) for contour in contours]


def _scale_contours(contours: list[np.ndarray], scale: tuple[float, float]):
    """Scale contour x/y to image x/y coordinates."""
    for i in range(len(contours)):
        c = contours[i]
        c[:, 0] *= scale[0]
        c[:, 1] *= scale[1]
        contours[i] = c


def _iter_padded_masks(data: np.ndarray, model: str) -> Iterator[np.ndarray]:
    """Yield the masks one at a time, padded by 1 pixel.

    Maskformer produces a 2D array of segment ids. Its segments are numbered from 1;
    0 marks pixels that belong to no segment and -1 fills an image without any
    segments, so those are skipped.
    """
    if model == "maskformer":
        padded = np.pad(data, 1)
        for segment_id in np.unique(data):
            if segment_id > 0:
                yield np.equal(padded, segment_id)
    else:
        for mask in data:
            yield np.pad(mask, 1)


def mask2poly(
    data: np.ndarray,
    model: Literal["maskformer", "dinosam", "bfms"],
    image: np.ndarray | None = None,
    tolerance: float | None = None,
) -> GeometryCollection:
    """Convert segmentation masks to a collection of (multi-)polygons.

    Args:
        data: Raw numpy array produced by segmentation routine (2D or 3D).
        model: Name of the model used for segmentation.
        image: Used for BFMS masks; bfms generates lower resolution masks, original
            image data is needed to rescale back.
        tolerance (optional): Tolerance value (in pixels) used to simpily geometry.
            Simplifying the geometry might be necessary when the polygons otherwise
            get too big and complex. Defaults to None.

    Returns:
        GeometryCollection: collection of all segmentations as multipolygons.
    """
    scale: tuple[float, float] | None = None

    if model not in ["maskformer", "dinosam", "bfms"]:
        msg = f"Invalid segmentation, model '{model}' is not supported."
        raise NotImplementedError(msg)

    if model == "bfms":
        if image is None:
            msg = "bfms generates lower resolution masks, image needed to rescale back."
            raise ValueError(msg)
        data_shape = data.shape
        if len(data_shape) == 3:
            data_shape = data_shape[1:]
        scale = (image.shape[0] / data_shape[0], image.shape[1] / data_shape[1])

    geometries = []
    for mask in _iter_padded_masks(data, model):
        contours = find_contours(mask, level=0.5)
        contours = _norm_contours(contours)  # remove padding again
        if scale is not None:
            _scale_contours(contours, scale)
        polys = [geometry.Polygon(contour) for contour in contours]
        geometries.append(
            geometry.MultiPolygon(
                polys
                if tolerance is None
                else [poly.simplify(tolerance) for poly in polys]  # type: ignore[misc]
            )
        )

    return GeometryCollection(geometries)
