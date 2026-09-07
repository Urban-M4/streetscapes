"""Plotting utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Delay slow imports for CLI responsiveness
    import geopandas as gpd


def make_colourmap(
    labels: dict | list | tuple,
    cmap: str = "jet",
) -> dict:
    """Create a dictionary of colours (used for visualising instances).

    Args:
        labels:
            A dictionary of labels.

        cmap:
            Colourmap. Defaults to "jet".

    Returns:
        dict:
            Dictionary of class/colour associations.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(labels) == 0:
        return {}

    cm = plt.get_cmap(cmap, len(labels))
    cm = cm(np.linspace(0.0, 1.0, cm.N))[:, :3]  # type: ignore
    return dict(zip(sorted(labels), cm, strict=False))  # type: ignore


def plot_metadata(gdf: "gpd.GeoDataFrame", ax=None):
    """Plot the metadata from a GeoDataFrame.

    Args:
        gdf:
            The GeoDataFrame containing the metadata.
        ax:
            The axes to plot on. Defaults to None.

    Returns:
        The axes with the plotted metadata.

    """
    import contextily as ctx

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(ax=ax, color="red", markersize=0.5, alpha=0.5)
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.nlmaps.standaard)
    return ax
