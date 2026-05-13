"""Segmentation mask utils."""

from typing import Literal

import numpy as np
from shapely import GeometryCollection, geometry
from skimage.measure import find_contours


def _norm_contours(contours: list[np.ndarray]):
    """Normalize contours to remove effect of padding."""
    return [np.maximum(contour - 1, 0) for contour in contours]


def _scale_contours(contours: list[np.ndarray], scale: tuple[float, float]):
    """Scale contour x/y to image x/y coordinates."""
    for i in range(len(contours)):
        c = contours[i]
        c[:,0] *= scale[0]
        c[:,1] *= scale[1]
        contours[i] = c


def _get_boolean_masks(data: np.ndarray):
    """Convert Maskformer 2D array to 3D array with boolean masks."""
    classes = np.unique(data)
    return np.stack([np.equal(data, _class) for _class in classes], axis=0)


def mask2poly(
    data: np.ndarray,
    model: Literal["maskformer", "dinosam", "bfms"],
    image: np.ndarray | None = None,
    tolerance: float | None = None
) -> GeometryCollection:
    """Convert segmentation masks to a collection of (multi-)polygons.

    Args:
        data: Raw numpy array produced by segmentation routine (2D or 3D).
        model: Name of the model used for segmentation.
        tolerance (optional): Tolerance value (in pixels) used to simpily geometry.
            Simplifying the geometry might be necessary when the polygons otherwise
            get too big and complex. Defaults to None.

    Returns:
        GeometryCollection: collection of all segmentations as multipolygons.   
    """
    scale: tuple[float, float] | None = None

    if not model in ["maskformer", "dinosam", "bfms"]:
        msg = f"Invalid segmentation, model '{model}' is not supported."
        raise NotImplementedError(msg)

    if model == "maskformer":
        data = _get_boolean_masks(data)

    if model == "bfms":
        if image is None:
            msg = "bfms generates lower resolution masks, image needed to rescale back."
            raise ValueError(msg)
        data_shape = data.shape
        if len(data_shape) == 3:
            data_shape = data_shape[1:]
        scale = (image.shape[0]/data_shape[0], image.shape[1]/data_shape[1])

    geometries = []
    for i in range(data.shape[0]):
        contours = find_contours(np.pad(data[i], 1), level=0.5)
        contours = _norm_contours(contours)  # remove padding again
        if scale is not None:
            _scale_contours(contours, scale)
        polys = [geometry.Polygon(contour) for contour in contours]
        geometries.append(
            geometry.MultiPolygon(polys if tolerance is None else [poly.simplify(tolerance) for poly in polys])  # type: ignore[misc]
        )

    return GeometryCollection(geometries)
