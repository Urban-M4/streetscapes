import typing as tp
from functools import reduce
from pathlib import Path

import ibis
import numpy as np

from streetscapes import utils
from streetscapes.streetview.instance import SVInstance


class SVSegmentation:
    """TODO: Add docstrings"""

    def __init__(
        self,
        path: Path | str,
    ):
        """
        A class that acts as an interface to an image segmentation
        and the object instances that are part of it.

        Args:

            path:
                Path to the segmentation archive.
        """

        # Path to the segmentation file
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"The segmentation mask for {self.path.name} does not exist."
            )

        # The model that was used to create this segmentation
        self.model = self.path.parent.name

        # Cache
        self._image: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self._instances: dict | None = None
        self._metadata: dict | None = None

        # Path to the image file
        # TODO: Handle different image types (resp. extensions).
        image_name = self._get_value("image_name")
        self.image_path = path.parent.parent.parent / image_name

    def __repr__(self):
        return f"SVSegmentation(path={utils.hide_home(self.path)!r}"

    def _get_value(self, key: str) -> tp.Any:
        """
        Retrieve a value for a given key from the segmentation file.

        Args:
            key: The key to load from the parquet file.

        Returns:
            The extracted value.
        """

        if self._metadata is None:
            self._metadata = np.load(self.path, allow_pickle=True)["arr_0"].item()
        return self._metadata[key]

    def _remove_overlaps(
        self, instances: dict[int, np.ndarray], exclude: str | list[str]
    ) -> np.ndarray:
        if isinstance(exclude, str):
            exclude = [exclude]

        image = self.get_image()
        canvas = np.zeros(image.shape[:2], dtype=np.uint32)

        for iid, mask in instances.items():
            canvas[mask] = iid

        for label in exclude:
            overlapping_instances = self.get_instances(label)
            if overlapping_instances:
                for instance in overlapping_instances:
                    canvas[instance.mask] = 0

        return {iid: (canvas == iid) for iid in instances}

    def get_image(
        self,
        cache: bool = False,
    ) -> np.ndarray:
        """
        Loads the image from the stored path.

        Args:
            cache:
                A toggle to indicate whether the image should be cached
                for slightly faster retrieval.
                Defaults to True.

        Returns:
            The image as a NumPy array.
        """

        if cache:
            if self._image is None:
                self._image = utils.open_image(self.image_path)
            return self._image

        return utils.open_image(self.image_path)

    def get_masks(self, cache: bool = False) -> np.ndarray:
        if cache and self._masks is not None:
            return self._masks

        masks = {iid: arr.astype(bool) for iid, arr in self._get_value("masks")}

        if cache:
            self._masks = masks

        return masks

    def get_instance_labels(
        self,
        cache: bool = False,
        as_table: bool = False,
    ) -> dict | ibis.Table:
        """
        Load the saved instances with their labels.

        Args:
            cache:
                A toggle to indicate whether the instances should be cached
                for slightly faster retrieval.
                Defaults to True.

            as_table:
                If this is True, the instance labels will be returned
                as an Ibis table instead of a dictionary.
                Defaults to False.

        Returns:
            The instances as a dictionary.
        """

        # Ensure that we have a
        if cache and self._instances is not None:
            instances = self._instances

        else:
            instances = dict(self._get_value("instances"))

        if cache:
            self._instances = instances

        if as_table:
            return ibis.memtable(
                [
                    {
                        "id": k,
                        "label": v,
                    }
                    for k, v in instances.items()
                ]
            )

        return instances

    def get_instances(
        self,
        label: str,
        exclude: str | list[str] | None = None,
        merge: bool = False,
    ) -> list[SVInstance]:
        """
        Return an array of instances corresponding to label.

        Args:
            label:
                Extract the instances for a given label.

            exclude:
                Mask labels to exclude. Defaults to None.

        Returns:
            The (potentially merged and de-overlapped) instances for this label.
        """

        instance_ids = set(
            [k for k, v in self.get_instance_labels().items() if v == label]
        )
        masks = {
            iid: mask
            for iid, mask in self.get_masks(cache=False).items()
            if iid in instance_ids
        }

        if not len(masks):
            return None

        if exclude is not None:
            masks = self._remove_overlaps(masks, exclude)

        if masks and merge:
            masks = {0: reduce(np.logical_or, masks.values())}

        instances = [
            SVInstance(self.image_path, mask, label, iid) for iid, mask in masks.items()
        ]

        return instances[0] if merge else instances

    def visualise(
        self,
        labels: str | list[str] | None = None,
        opacity: float = 0.5,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
    ) -> tuple:
        """
        Visualise the instances of different objects in an image.

        Args:

            labels:
                Labels for the instance categories that should be plotted.

            opacity:
                Opacity to use for the segmentation overlay.
                Defaults to 0.5.

            title:
                The figure title.
                Defaults to None.

            figsize:
                Figure size. Defaults to (16, 6).

        Returns:
            A tuple containing:
                - A Figure object.
                - An Axes object that allows further annotations to be added.
        """
        from matplotlib import patches as mpatches
        from matplotlib import pyplot as plt

        # Prepare the greyscale version of the image for plotting instances.
        image = self.get_image()
        greyscale = utils.as_rgb(image, greyscale=True)

        # Load the masks
        masks = self.get_masks()

        # Create a figure
        (fig, axes) = plt.subplots(1, 2, figsize=figsize)

        # Load the instance labels
        instances = self.get_instance_labels()

        if labels is None:
            labels = set(instances.values())

        elif isinstance(labels, str):
            labels = [labels]

        # Prepare the colour dictionary and the layers
        # necessary for plotting the category patches.
        colourmap = utils.make_colourmap(labels)

        # Label handles for the plot legend.
        handles = {}

        # Loop over the segmentation list
        for instance_id, label in instances.items():
            if label not in labels:
                # Skip labels that have been removed.
                continue

            if label not in handles:
                # Add a basic coloured label to the legend
                handles[label] = mpatches.Patch(
                    color=colourmap[label],
                    label=label,
                )

            # Extract the mask
            mask = masks[instance_id]
            if mask.sum() == 0:
                continue

            greyscale[mask] = (
                (1 - opacity) * greyscale[mask] + 255 * opacity * colourmap[label]
            ).astype(np.ubyte)

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

        fig.tight_layout()
        return (fig, axes)
