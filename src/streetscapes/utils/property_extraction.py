from typing import Optional

import matplotlib
import matplotlib.axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
import shapely
from PIL.Image import Image


def poly_to_mask(
    poly: shapely.Polygon | shapely.MultiPolygon,
    img: Image
) -> np.ndarray:
    """Convert a segmentation polygon to an image mask.

    Args:
        poly: The (multi)polygon to rasterize
        img: The image that the segmentation polygon was based on.

    Returns:
        2-D numpy array containing the image mask.
    """
    h, w, _ = np.asarray(img).shape
    return rasterio.features.rasterize(  # type: ignore[no-any-return]
        [poly], out_shape=(h,w),
    )


def plot_multipolygon(poly: shapely.MultiPolygon, ax: Optional[matplotlib.axes.Axes] = None, color = "r"):
    """Plot a MultiPolygon with Matplotlib, e.g. to overlay on an image.

    Args:
        poly: MultiPolygon to plot
        ax (optional): Matplotlib axis to plot the polygon into. Defaults to the current
            active matplotlib plot.
        color (optional): Matplotlib color name to color the polygon.
    """
    for geom in poly.geoms:    
        xs, ys = geom.exterior.xy
        if ax is None:
            ax = plt  # type: ignore
        ax.fill(xs, ys, alpha=0.5, fc=color, ec="none")  # type: ignore


def mask_image(img: Image, mask: np.ndarray) -> np.ma.MaskedArray:
    """Mask an Image object with a binary mask

    Note: numpy masks away values under the mask. As we're interested in the values
    under the mask we need to invert the mask her.

    Returns:
        Masked numpy array
    """
    return np.ma.masked_array(img, (1 - mask).repeat(3, axis=np.newaxis), ndmin=2)


def display_color(rgb: tuple[float, float, float]) -> None:
    """Display an RGB color as a rectangle, for debugging and testing."""
    rgb = tuple([c / 255 for c in rgb])  # type: ignore
    rect = patches.Rectangle((0, 0), 1, 1, edgecolor="none", facecolor=rgb)
    _, ax = plt.subplots(1,1, figsize=(1,1))
    ax.add_patch(rect)
    ax.set_axis_off()
