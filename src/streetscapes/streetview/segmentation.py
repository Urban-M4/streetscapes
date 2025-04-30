# --------------------------------------
from pathlib import Path

# --------------------------------------
from PIL import Image

# --------------------------------------
import ibis

# --------------------------------------
import numpy as np

# --------------------------------------
from streetscapes import utils
from streetscapes.utils import logger
from streetscapes.models import ModelType
from streetscapes.streetview.instance import SVInstance


class SVSegmentation:
    """TODO: Add docstrings"""
    @staticmethod
    def from_saved(
        path: Path,
    ):
        """Load mask and labels from previously saved segmentation."""

        mask_file = path.with_suffix("npz")
        label_file = path.with_suffix("stat.parquet")

        try:
            mask = np.load(mask_file, allow_pickle=False)["arr_0"]
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Mask file not found. Have you segmented the image?"
            ) from exc

        try:
            stats = ibis.read_parquet(label_file).loc[0]
            labels = dict(zip(stats.instance.tolist(), stats.label))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Label file not found. Have you segmented the image with stats calculation?"
            ) from exc

        return SVSegmentation(mask, labels)

    def __init__(
        self,
        model: ModelType | str,
        image_path: Path | str,
    ):
        '''
        A class that acts as an interface to an image segmentation
        and the object instances that are part of it.

        Args:
            model:
                The model used for segmenting the image.

            image_path:
                The path to the image.
        '''
        self.model = ModelType(model) if isinstance(model, str) else model
        self.image_path = image_path

        # Extract the mask and instance paths
        # ==================================================
        root_dir = self.image_path.parent
        name = f"{self.model.name.lower()}/{self.image_path.name}"
        self.mask_path = utils.make_path(f"masks/{name}", root_dir, "npz")
        self.instance_path = utils.make_path(f"instances/{name}", root_dir, "parquet")

        # Cache
        self._mask: np.ndarray | None = None
        self._instances: ibis.Table | None = None

    def __repr__(self):
        return f"Segmentation(mask={self.mask_path}"

    def get_image(self) -> np.ndarray:
        """
        Loads the image from the stored path.

        Returns:
            The image as a NumPy array.
        """
        return np.asarray(Image.open(self.image_path))

    def get_mask(
        self,
        cache: bool = False,
    ) -> np.ndarray | None:
        """
        Load the saved mask as a NumPy array.

        Args:
            cache:
                A toggle to indicate whether the mask should be cached
                for slightly faster retrieval. Defaults to False.

        Returns:
            The mask as a NumPy array (if it exists).
        """

        # Ensure that we have a list of path objects
        if self._mask is not None:
            return self._mask

        if self.mask_path is None or not self.mask_path.exists():
            raise FileNotFoundError(
                "The mask does not exist. Have you segmented the image?"
            )

        mask = np.load(self.mask_path, allow_pickle=False)["arr_0"]

        if cache:
            self._mask = mask

        return mask

    def get_instance_table(
        self,
        cache: bool = False,
    ) -> ibis.Table | None:
        """
        Load the saved instances as an Ibis table.

        Args:
            cache:
                A toggle to indicate whether the instances should be cached
                for slightly faster retrieval. Defaults to False.

        Returns:
            The instances as an Ibis table.
        """

        # Ensure that we have a
        if self._instances is not None:
            return self._instances

        if self.instance_path is None or not self.instance_path.exists():
            raise FileNotFoundError(
                "The instance path not exist. Have you segmented the image?"
            )

        instances = ibis.read_parquet(self.instance_path)

        if cache:
            self._instances = instances

        return instances

    def get_instances(
        self,
        label: str,
    ) -> list[SVInstance]:
        """
        Return an array of instances corresponding to label.
        """
        instance_ids = [k for _, (k, v) in self.get_instance_table().to_pandas().iterrows() if v == label]
        mask = self.get_mask()
        if mask is None:
            return []
        instance_masks = [mask == index for index in instance_ids]
        return [SVInstance(mask, label) for mask in instance_masks]

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
        from matplotlib import pyplot as plt
        from matplotlib import patches as mpatches
        from matplotlib.colors import hex2color

        # Prepare the greyscale version of the image for plotting instances.
        image = self.get_image()
        greyscale = utils.as_rgb(image, greyscale=True)

        # Load the mask
        mask = self.get_mask()

        # Create a figure
        (fig, axes) = plt.subplots(1, 2, figsize=figsize)

        # Load the instances
        instances = self.get_instance_table()

        if labels is None:
            labels = instances.distinct().select("label").to_pandas().to_numpy()[:, 0]

        elif isinstance(labels, str):
            labels = [labels]

        # Prepare the colour dictionary and the layers
        # necessary for plotting the category patches.
        colourmap = utils.make_colourmap(labels)

        # Label handles for the plot legend.
        handles = {}

        # Loop over the segmentation list
        for row, (instance_id, label) in instances.to_pandas().iterrows():

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
            inst_mask = mask == instance_id
            if not np.any(inst_mask):
                continue

            greyscale[inst_mask] = (
                (1 - opacity) * greyscale[inst_mask] + 255 * opacity * colourmap[label]
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
