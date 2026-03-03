"""Segmentation mask utils."""

from typing import Literal, Optional

import numpy as np
from shapely import GeometryCollection, geometry
from skimage.measure import find_contours


def _norm_contours(contours: list[np.ndarray]):
    """Normalize contours to remove effect of padding."""
    return [np.maximum(contour - 1, 0) for contour in contours]


def _get_boolean_masks(data: np.ndarray):
    """Convert Maskformer 2D array to 3D array with boolean masks."""
    classes = np.unique(data)
    return np.stack([np.equal(data, _class) for _class in classes], axis=0)


def mask2poly(
    data: np.ndarray,
    model: Literal["maskformer", "dinosam", "bfms"],
    tolerance: Optional[float] = None
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
    if not model in ["maskformer", "dinosam"]:
        msg = f"Invalid segmentation, model '{model}' is not supported at the moment."
        raise NotImplementedError(msg)

    if model == "maskformer":
        data = _get_boolean_masks(data)

    geometries = []
    for i in range(data.shape[0]):
        contours = find_contours(np.pad(data[i], 1), level=0.5)
        contours = _norm_contours(contours)  # remove padding again
        polys = [geometry.Polygon(contour) for contour in contours]
        if tolerance is not None:
            polys = [poly.simplify(tolerance) for poly in polys]
        poly = geometry.MultiPolygon(polys)
        geometries.append(poly)

    return GeometryCollection(geometries)
