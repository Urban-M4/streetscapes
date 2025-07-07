# --------------------------------------
from pathlib import Path

# --------------------------------------
import torch
from torch import nn

# --------------------------------------
import torchvision.transforms as tvt

# --------------------------------------
import ibis

# --------------------------------------
import re

# --------------------------------------
import numpy as np

# --------------------------------------
import typing as tp

# --------------------------------------
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
        self._materials: dict | None = None
        self._metadata: dict | None = None

        # Path to the image file
        # TODO: Handle different image types (resp. extensions).
        image_name = self._get_value("image_name")
        self.image_path = path.parent.parent.parent / image_name

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(path={utils.hide_home(self.path)!r}"

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
        self,
        instances: dict[int : np.ndarray],
        exclude: str | list[str],
    ) -> np.ndarray:
        """
        Remove overlaps between labels.

        Args:

            labels:
                Masks corresponding to instances of interest.

            excluded:
                One or more labels for instances that should be excluded.

        Returns:
            The masks with undesired overlapping instances removed.
        """

        if isinstance(exclude, str):
            exclude = [exclude]

        # The original image
        image = self.get_image()

        # Make new positive and negative blank mask canvases
        canvas = np.zeros(image.shape[:2], dtype=np.uint32)

        # Mask with all instances
        for iid, mask in instances.items():
            canvas[mask[0], mask[1]] = iid

        # Remove overlaps
        excluded = self.get_instances(exclude, merge=True)

        for label, instance in excluded.items():
            if instance.indices is not None:
                canvas[instance.indices[0], instance.indices[1]] = 0

        return {iid: np.where(canvas == iid) for iid in instances}

    def get_image(
        self,
        cache: bool = False,
        refresh: bool = False,
        greyscale: bool = False,
    ) -> np.ndarray:
        """
        Loads the image from the stored path.

        Args:
            cache:
                A toggle to indicate whether the image should be cached
                for slightly faster retrieval.

            refresh:
                If set, extract the materials and refresh the cache with the new results.

        Returns:
            The image as a NumPy array.
        """

        if cache and (self._image is not None) and (not refresh):
            return self._image

        image = utils.as_rgb(utils.open_image(self.image_path), greyscale=greyscale)

        if refresh or (cache and self._image is None):
            self._image = image

        return image

    def get_masks(
        self,
        cache: bool = False,
        refresh: bool = False,
    ) -> np.ndarray:
        """
        Load the saved masks as a NumPy array.

        Args:
            cache:
                A toggle to indicate whether the mask should be cached
                for slightly faster retrieval at the expense of higher memory usage.

            refresh:
                If set, extract the materials and refresh the cache with the new results.

        Returns:
            The mask as a NumPy array (if it exists).
        """

        # Ensure that we have a list of path objects
        if cache and (self._masks is not None) and (not refresh):
            return self._masks

        masks = {
            iid: np.array(arr, dtype=np.uint32) for iid, arr in self._get_value("masks")
        }

        if refresh or (cache and self._masks is None):
            self._masks = masks

        return masks

    def get_instance_labels(
        self,
        cache: bool = False,
        refresh: bool = False,
        as_table: bool = False,
    ) -> dict | ibis.Table:
        """
        Load the saved instances with their labels.

        TODO: Fix the logic around `as_table` because it is not going to work
        in case we are retrieving data from the cache.

        Args:
            cache:
                A toggle to indicate whether the instances should be cached
                for slightly faster retrieval at the expense of higher memory usage.

            refresh:
                If set, extract the materials and refresh the cache with the new results.

            as_table:
                If this is True, the instance labels will be returned
                as an Ibis table instead of a dictionary.

        Returns:
            The instances as a dictionary or an Ibis table.
        """

        # Ensure that we have a
        if cache and (self._instances is not None) and (not refresh):
            return self._instances

        instances = dict(self._get_value("instances"))

        if as_table:
            instances = ibis.memtable(
                [
                    {
                        "id": k,
                        "label": v,
                    }
                    for k, v in instances.items()
                ]
            )

        if refresh or (cache and self._instances is None):
            self._instances = instances

        return instances

    def get_instances(
        self,
        labels: str | list[str],
        exclude: str | list[str] | None = None,
        merge: bool = False,
    ) -> SVInstance | dict[str, SVInstance]:
        """
        Return a dictionary of labels mapped to instances.

        Args:
            labels:
                One or more labels.

            exclude:
                Mask labels to exclude.

            merge:
                Merge all instances of each label into one.

        Returns:
            The (potentially merged and de-overlapped) instances for this label.
        """
        instances = {}
        single = False
        if isinstance(labels, str):
            labels = [labels]
            single = True

        for label in labels:
            instance_ids = set(
                [k for k, v in self.get_instance_labels().items() if v == label]
            )

            masks = {
                iid: mask
                for iid, mask in self.get_masks().items()
                if iid in instance_ids
            }

            if exclude is not None:
                masks = self._remove_overlaps(masks, exclude)

            if merge:
                merged = (
                    None
                    if len(masks) == 0
                    else np.concatenate(list(masks.values()), axis=1)
                )
                masks = {0: merged}

            instance_objects = [
                SVInstance(self.image_path, mask, label, iid)
                for iid, mask in masks.items()
            ]
            instances[label] = instance_objects[0] if merge else instance_objects

        return instances[labels[0]] if single else instances

    def visualise_instances(
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
        image = self.get_image()
        greyscale = self.get_image(greyscale=True)

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
            if mask.size == 0:
                continue

            greyscale[mask[0], mask[1]] = (
                (1 - opacity) * greyscale[mask[0], mask[1]]
                + 255 * opacity * colourmap[label]
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

        return (fig, axes)

    def get_materials(
        self,
        model: nn.Module,
        labels: str | list[str] | None = None,
        cache: bool = False,
        refresh: bool = False,
    ):
        """
        Extract materials from an image provided as a NumPy array.

        Args:
            model:
                The model to use for the material segmentation.

            labels:
                A list of labels to consider.

            cache:
                If set, cache the results.

            refresh:
                If set, extract the materials and refresh the cache with the new results.
        """

        if cache and (self._materials is not None) and (not refresh):
            return self._materials

        if labels is None:
            labels = list(model.taxonomy.values())

        elif isinstance(labels, str):
            labels = [labels]

        image = self.get_image()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = tvt.Normalize(
            image_tensor.mean(dim=(1, 2)), image_tensor.mean(dim=(1, 2))
        )(image_tensor)[None, ...]

        with torch.no_grad():
            materials = model(image_tensor)[0].squeeze().numpy()

        if refresh or (cache and self._materials is None):
            self._materials = materials

        m_ids = np.unique(materials, axis=None)

        taxonomy = {m_id: m_label for m_id, m_label in model.taxonomy.items() if m_id in m_ids}

        return materials, taxonomy

    def visualise_materials(
        self,
        materials: np.ndarray,
        taxonomy: dict,
        labels: str | list[str] | None = None,
        opacity: float = 0.5,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
    ) -> tuple:
        """
        Visualise the instances of different objects in an image.

        Args:

            materials:
                The results obtained from a material segmentation model.

            taxonomy:
                The label taxonomy to use (label ID: label).

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

        image = self.get_image()
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
