# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import torch

# --------------------------------------
import torchvision.transforms as tvt

# --------------------------------------
import re

# --------------------------------------
from collections.abc import Callable

# --------------------------------------
from typing import Any

# --------------------------------------
from streetscapes import utils


class SVInstance:
    """TODO: Add docstrings"""

    def __init__(
        self,
        image_path: Path,
        indices: np.ndarray,
        label: str,
        iid: int,
    ):
        self.image_path = image_path
        self.indices = indices
        self.label = label
        self.iid = iid

    @property
    def mask(self):
        _mask = np.zeros_like(
            utils.open_image(self.image_path, greyscale=True), dtype=np.bool_
        )
        if self.indices is not None:
            _mask[self.indices[0], self.indices[1]] = True

        return _mask

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(label={self.label})"

    def masked(
        self,
        channel: int = None,
        hsv: bool = False,
    ) -> np.ndarray:
        """
        Masked pixels corresponding to the instance.

        Args:
            channel:
                Only get the masked data for a specific channel.
                Defaults to None.

            hsv:
                Convert the original RBG image into HSV.
                Defaults to None.

        Returns:
            The pixels corresponding to the mask.
        """

        image = utils.open_image(self.image_path)
        if hsv:
            image = utils.as_hsv(image)
        masked = np.zeros_like(image)

        if channel is None:
            channel = slice(None, None)

        masked[self.mask, channel] = image[self.mask, channel]

        return masked

    def brightness(self) -> np.ndarray:
        """
        A convenience method to extract the brightness (= the value in HSV space)
        for the pixels corresponding to this instance.

        Returns:
            The per-pixel brightness of the instance.
        """

        return self.masked(hsv=True, channel=2)

    def visualise(
        self,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
        channel: int = None,
    ):
        """
        Visualise this instance only, with the rest of the image blanked out.

        Args:
            title: Add a title to the plot.
            figsize: Figure size.
            channel: Only plot the requested channel.

        Returns:
            The instance plotted in isolation.
        """
        import matplotlib.pyplot as plt

        # Create a figure
        (fig, axes) = plt.subplots(1, 1, figsize=figsize)

        canvas = self.masked(channel)

        axes.imshow(canvas)
        axes.axis("off")
        if title is not None:
            fig.suptitle(title, fontsize=16)

        return (fig, axes)

    def apply(self, fn: Callable) -> float | np.ndarray:
        return fn(self.masked())

    def get_materials(
        self,
        model: Any,
        cache: bool = False,
        refresh: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """
        Extract materials from this instance.

        Args:
            model:
                A material recognition model.

            cache:
                If set, cache the results.

            refresh:
                If set, extract the materials and refresh the cache with the new results.

        Returns:
            The material label predictions.
        """

        image = self.masked()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = tvt.Normalize(
            image_tensor.mean(dim=(1, 2)), image_tensor.mean(dim=(1, 2))
        )(image_tensor)[None, ...]

        with torch.no_grad():
            materials = model(image_tensor)[0].squeeze().numpy()

        materials[~self.mask] = 0

        return materials

    def visualise_materials(
        self,
        taxonomy: dict,
        materials: dict,
        labels: str | list[str] | None = None,
        opacity: float = 0.5,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
    ) -> tuple:
        """
        Visualise the instances of different objects in an image.

        Args:

            taxonomy:
                The label taxonomy to use (label ID: label).

            materials:
                The results obtained from a material segmentation model.

            labels:
                Labels for the instance categories that should be plotted.

            opacity:
                Opacity to use for the segmentation overlay.

            title:
                The figure title.

            figsize:
                Figure size.

        Returns:
            A tuple containing:
                - A Figure object.
                - An Axes object that allows further annotations to be added.
        """
        from matplotlib import pyplot as plt
        from matplotlib import patches as mpatches

        # Prepare the greyscale version of the image for plotting instances.

        image = self.masked()
        greyscale = utils.as_rgb(image, greyscale=True)

        # Create a figure
        (fig, axes) = plt.subplots(1, 2, figsize=figsize)

        if labels is None:
            labels = set(taxonomy.values())

        elif isinstance(labels, str):
            labels = [labels]

        inverse_taxonomy = {re.sub(r"\s+", " ", v): k for k, v in taxonomy.items()}

        # Extract the label IDs
        label_ids = set()
        for label in labels:
            label = re.sub(r"\s+", " ", label.lower()).strip()
            label_id = inverse_taxonomy.get(label)
            if label_id is not None:
                label_ids.add(label_id)

        # Prepare the colour dictionary and the layers
        # necessary for plotting the category patches.
        colourmap = utils.make_colourmap(labels)

        # Label handles for the plot legend.
        handles = {}

        for label_id in label_ids:

            if label_id not in taxonomy:
                # Skip labels that have been removed.
                continue

            label = taxonomy.get(label_id)

            if label is None:
                continue

            if label not in handles:
                # Add a basic coloured label to the legend
                handles[label] = mpatches.Patch(
                    color=colourmap[label],
                    label=label,
                )

            # Extract the mask
            mask = materials == label_id

            if mask.size == 0:
                continue

            greyscale[mask] = (1 - opacity) * greyscale[
                mask
            ] + 255 * opacity * colourmap[label]

        # Plot the original image and the segmented one.
        # If any of the requested categories exist in the
        # image, they will be overlaid as coloured patches
        # with the given opacity over the original image.
        # ==================================================
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(greyscale)
        axes[1].axis("off")
        axes[1].legend(
            handles=handles.values(), loc="upper left", bbox_to_anchor=(1, 1)
        )
        if title is not None:
            fig.suptitle(title, fontsize=16)

        return (fig, axes)
