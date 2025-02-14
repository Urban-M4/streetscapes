# --------------------------------------
import re

# --------------------------------------
from pathlib import Path

# --------------------------------------
import PIL
from PIL import Image

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------
import PIL.ImageFile
from tqdm import tqdm

# --------------------------------------
import numpy as np

# --------------------------------------
import scipy

# --------------------------------------
import torch as pt

# --------------------------------------
import pandas as pd

# --------------------------------------
import awkward as ak

# --------------------------------------
import skimage as ski

# --------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------------------
from streetscapes import conf
from streetscapes.conf import logger


ImageOrPath = (
    Path
    | str
    | np.ndarray
    | Image.Image
    | list[Path | str | np.ndarray | Image.Image]
    | tuple[Path | str | np.ndarray | Image.Image]
)


class BaseSegmenter:

    # Dictionary of label ID to label
    labels = {}

    @property
    def label_ids(self):
        return {label: label_id for label_id, label in self.labels.items()}

    def __init__(
        self,
        device: str = None,
    ):
        """
        A model serving as the base for all segmentation models.

        Args:
            device (str, optional):
                Device to use for processign. Defaults to None.

        """
        # Set up the device
        if device is None:
            device = "cuda" if pt.cuda.is_available() else "cpu"
        self.device = pt.device(device)

    def _load_models(self, *args, **kwargs):
        """
        Convenience function for loading a pretrained model.

        Raises:
            NotImplementedError:
                An error if this method is not overridden.
        """
        raise NotImplementedError("Please implement this method in a derived class")

    def segment(self, *args, **kwargs):
        """
        Convenience function for segmenting images.

        Raises:
            NotImplementedError:
                An error if this method is not overridden.
        """
        raise NotImplementedError("Please implement this method in a derived class")

    def load_stats(
        self,
        stats: str | Path,
    ) -> ak.Array:
        """
        Load statistics from a Parquet file.

        Args:
            stats (str | Path):
                The path to the Parquet file.

        Returns:
            ak.Array:
                An Awkward array
        """

        # Load the data
        data = {
            item["image"]: item["stats"].tolist()
            for item in list(ak.from_parquet(stats))
        }

        # Fix for empty lists appearing as None
        # FIXME: Find a more generic solution.
        for image_name, categories in data.items():
            for category, stats in categories.items():
                if stats is None:
                    categories[category] = []

        return data

    def save_stats(
        self,
        stats: dict,
        filename: Path | str,
        out_dir: Path | str = None,
    ) -> Path:
        """
        Save image statistics as an Awkward array to a Parquet file.

        Args:

            filename (Path | str):
                Name of the output file.

            out_dir (Path | str, optional):
                Directory where the output file should be saved. Defaults to None.

        Returns:
            Path:
                The path to the saved file.
        """

        if out_dir is None:
            out_dir = conf.OUTPUT_DIR

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / filename

        stats = [
            {
                "image": image_name,
                "stats": image_stats,
            }
            for image_name, image_stats in stats.items()
        ]

        # Save the file
        ak.to_parquet(stats, out_file)

        return out_file

    def load_images(
        self,
        images: ImageOrPath,
    ) -> dict[str, np.ndarray]:
        """
        A list of images or paths to image files.

        Args:
            images (ImagePath):
                A list of paths to image files.

        Returns:
            dict[str, np.ndarray]:
                A dictionary of image names mapped to images as NumPy arrays.
        """

        names = None
        if isinstance(images, dict):
            names = list(images.keys())
            images = list(images.values())

        if not isinstance(images, (list, tuple)):
            images = [images]

        image_dict = {}

        if names is None:
            names = [f"image_{n:04d}" for n in range(len(images))]

        for idx, image in enumerate(images):
            if isinstance(image, (str, Path)):
                image_name = image.name
                image = Image.open(image)

            else:
                image_name = names[idx]

            image_dict[image_name] = np.array(image)

        return image_dict

    def make_colour_dict(
        self,
        categories: dict | list | tuple,
        opacity: float = 1.0,
        cmap: str = "jet",
    ) -> dict:
        """
        Create a dictionary of colours (used for visualising the segments).

        Args:
            categories (dict | list | tuple):
                A dictionary of categories.

            opacity (float, optional):
                Opacity to use for the segmentation patches.
                Should be between 0.0 and 1.0. Defaults to 1.0.

            cmap (str, optional):
                Matplotlib colourmap. Defaults to "jet".

        Returns:
            dict:
                Dictionary of class/colour associations.
        """

        cmap = plt.get_cmap(cmap)
        colours = {}
        for i, cls_id in enumerate(categories):
            colour = np.array(cmap(1.0 * i / (len(categories))))
            colour[-1] = 1
            colours[cls_id] = opacity * colour

        return colours

    def check_categories_against_labels(
        self,
        categories: list[str, int],
    ) -> set[str]:
        """
        Unify a set of categories that the user is interested in
        with the categories that are actually available.

        Args:
            categories (list[str, int]):
                Requested categories.

        Returns:
            set[str]:
                A set of categories that are guaranteed to exist.
        """

        # Make sure that the requested set is sane
        if categories is None:
            categories = self.labels.values()

        categories = set(
            [
                re.sub(
                    r"\s+",
                    "-",
                    (self.labels[cat] if isinstance(cat, int) else cat)
                    .lower()
                    .strip()
                    .replace("-", " "),
                )
                for cat in categories
            ]
        ).intersection(set(self.labels.values()))
        category_dict = {self.label_ids[cat]: cat for cat in categories}

        return (categories, category_dict)

    def compute_class_statistics(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> dict:
        """
        Compute statistics over a patch of the segmented image.

        Args:
            image (np.ndarray):
                The segmented image.

            mask (np.ndarray):
                A mask corresponding to a single class.

        Returns:
            dict:
                A dictionary of statistics.
        """

        # Rescale intensity to (0,1) (converts the image to float32)
        float_image = ski.exposure.rescale_intensity(image, out_range=(0, 1))
        rgb_patch = float_image[mask]

        # Convert the image to HSV
        hsv_image = ski.color.convert_colorspace(image, "RGB", "HSV")
        hsv_patch = hsv_image[mask]

        # Extract the HSV channels to compute the corresponding statistics.
        hue, sat, val = hsv_patch[..., 0], hsv_patch[..., 1], hsv_patch[..., 2]

        return {
            "colour": {
                "median": np.median(rgb_patch, axis=0).tolist(),
                "mode": scipy.stats.mode(rgb_patch, axis=0)[0].tolist(),
                "mean": np.mean(rgb_patch, axis=0).tolist(),
                "sd": np.std(rgb_patch, axis=0).tolist(),
            },
            "hue": {
                "median": np.median(hue).tolist(),
                "mode": scipy.stats.mode(hue)[0].tolist(),
                "mean": np.mean(hue).tolist(),
                "sd": np.std(hue).tolist(),
            },
            "saturation": {
                "median": np.median(sat).tolist(),
                "mode": scipy.stats.mode(sat)[0].tolist(),
                "mean": np.mean(sat).tolist(),
                "sd": np.std(sat).tolist(),
            },
            "value": {
                "median": np.median(val).tolist(),
                "mode": scipy.stats.mode(val)[0].tolist(),
                "mean": np.mean(val).tolist(),
                "sd": np.std(val).tolist(),
            },
            "area": float(np.count_nonzero(mask) / np.prod(image.shape[:2])),
        }

    def compute_statistics(
        self,
        images: dict,
    ) -> dict:
        """
        Compute the statistics for all categories.

        Args:
            segmentations (dict):
                A dictionary of segmentation results.

        Returns:
            dict:
                A dictinary with statistics about the image segments.
        """

        # Loop over the images
        logger.info("Computing statistics...")
        results = {}
        with tqdm(total=len(images)) as pbar:
            for image in images.items():
                instance_stats = {}
                # Loop over categories per image
                for label_id, instances in image["masks"].items():
                    if len(instances) > 0:
                        instance_stats[self.labels[label_id]] = []
                        # Loop over instances per category
                        for instance in instances:

                            # Compute the statistics for that instance
                            stats = self.compute_class_statistics(
                                image, instance["mask"]
                            )
                            instance_stats[self.labels[label_id]].append(stats)

                pbar.update()
        return results

    def visualise_segmentation(
        self,
        images: ImageOrPath,
        categories: list[str | int] = None,
        opacity: float = 0.50,
        figsize: tuple[int, int] = (16, 6),
        stats: ak.Array | list | str | Path = None,
    ) -> tuple[plt.Figure, plt.Axes, dict]:
        """
        Segment and visualise an image, and potentially show the colour statistics
        associated with the requested categories.

        Args:
            images (ImagePath):
                Images being visualised.

            categories (list[str | int], optional):
                Categories to extract. Defaults to None.

            opacity (float, optional):
                Opacity to use for the segmentation overlay. Defaults to 0.50.

            figsize (tuple[int, int], optional):
                Figure size. Defaults to (16, 8).

            stats (ak.Array | list | str | Path, optional):
                An `awkward` array of statistics. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes, dict]:
                A tuple containing:
                    - A Figure object;
                    - An Axes object;
                    - A set of statistics for the requested categories.
        """

        # Load the images
        images = self.load_images(images)

        # Segment the image(s).
        # ==================================================
        segmentations = self.segment(images)

        # Unify the styles of labels and categories
        # so that they can be compared.
        # ==================================================
        (categories, category_dict) = self.check_categories_against_labels(categories)

        # Adjust the figsize based on how many images are being plotted
        figsize = list(figsize)
        figsize[1] *= len(images)

        # Create a figure
        (fig, axes) = plt.subplots(len(images), 2, figsize=figsize)

        # Load the statistics if provided as a file.
        if isinstance(stats, (str, Path)):
            self.load_stats(stats)

        # Prepare the colour dictionary and the layers
        # necessary for plotting the category patches.
        # ==================================================
        image_stats = {}
        with tqdm(total=len(images)) as pbar:
            for idx, (image_name, image) in enumerate(images.items()):

                segmentation = segmentations[image_name]

                # Get the right axis in case there is more than one row
                ax = axes if len(images) == 1 else axes[idx]

                # Convert the image to float [0,1]
                image = ski.exposure.rescale_intensity(image, out_range=(0, 1))

                # Add an alpha channel if it's missing.
                # A greyscale image has one channel, and an RGB one has three.
                if len(image.shape) in [1, 3]:
                    opacity_channel = np.full(image.shape[:-1], 1)[:, :, None]
                    image_with_alpha = np.concatenate((image, opacity_channel), axis=-1)

                # The image opacity is complementary to the segment opacity
                image_opacity = 1 - opacity

                # Prepare the colour dictionary
                colours = self.make_colour_dict(category_dict, opacity)

                # Create an outline layer
                outlines = np.zeros(image.shape[:2])

                # Label handles for the plot legend.
                handles = []

                # Create a dictionary for the statistics for this image
                image_stats[image_name] = {}

                # Loop over the segmentation list
                for label_id, instances in segmentation["labels"].items():
                    if label_id in category_dict:

                        if len(instances) > 0:
                            # Add a basic class legend and extract the statistics
                            handles.append(
                                mpatches.Patch(
                                    color=colours[label_id][:-1] / opacity,
                                    label=category_dict[label_id],
                                )
                            )

                            # Loop over instances and add an overlay with
                            # their corresponding colours
                            for instance in instances:

                                mask = instance["mask"]
                                outlines[mask] = instance["id"]

                                image_with_alpha[mask] = (
                                    image_opacity * image_with_alpha[mask]
                                    + colours[label_id]
                                )

                        # Get the statistics for the requested categories only
                        if stats is not None:
                            cat = category_dict[label_id]
                            if cat in stats[image_name]:
                                image_stats[image_name][cat] = stats[image_name][cat]

                # Outline the instances (the colour is equivalent to "#00ff00ff")
                outlines = ski.segmentation.find_boundaries(outlines, mode="thick")
                image_with_alpha[outlines != 0] = np.array([0, 1, 0, 1])

                # Plot the original image and the segmented one.
                # If any of the requested categories exist in the
                # image, they will be overlaid as coloured patches
                # with the given opacity over the original image.
                # ==================================================
                ax[0].imshow(image)
                ax[0].axis("off")
                ax[1].imshow(image_with_alpha)
                ax[1].axis("off")
                ax[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
                ax[0].set_title(image_name)

                pbar.update()

        return (fig, ax, image_stats)

    def segment_from_dataset(
        self,
        path: Path | str,
        image_dir: Path | str = None,
        out_file_path: Path | str = None,
        sample: int = None,
    ) -> tuple[dict, Path | None]:
        """
        Segment images specified in a CSV data (sub)set.

        Args:
            path (Path | str):
                The input dataset (a CSV file).

            image_dir (Path | str):
                The directory where downloaded images are located. Defaults to None.

            out_file_path (Path | str, optional):
                Name of the output file. Defaults to None.

            sample (int, optional):
                The size of the sample (used for testing purposes).

        Returns:
            list[dict]:
                The statistics for all the segmented images.
        """

        # Load the CSV file.
        # ==================================================
        df = pd.read_csv(path)

        # Directory containing the downloaded images
        # ==================================================
        if image_dir is None:
            image_dir = conf.OUTPUT_DIR / "images"

        images = sorted(
            [image_dir / f"{file_name}.jpeg" for file_name in df["orig_id"]]
        )
        if sample is not None:
            images = images[:sample]

        # Load the images
        images = self.load_images(images)

        # Segment the images
        # ==================================================
        segmented = self.segment(images)

        # Compute the statistics
        # ==================================================
        stats = self.compute_statistics(images, segmented)

        # Save the dataset if a file name is provided
        # ==================================================
        if out_file_path is not None:
            self.save_stats(stats, out_file_path)

        return (images, segmented, stats)

    def flatten_categories(
        self,
        categories: dict,
    ) -> dict:
        """
        Flatten a nested dictionary of categories.

        Useful for defining masks in terms of more general categories
        that can be subtracted.
        For instance, building facades can include windows and doors,
        which means that we can define a category dictionary as follows:

        categories = {
            "sky": None,
            "building": {
                "window": None,
                "door": None
            },
            "tree": None,
            "car": None,
            "road": None,
        }

        The model will subtract the masks for `window` and `door` from
        that for `building`, so when the statistics are computed, only
        the portion of the building without windows and doors will be
        taken into acount.

        Args:
            tree (dict):
                The categories as a tree (dictionary of dictionaries).

        Returns:
            dict:
                A flattened category tree where each key is a
                category and the corresponding value is a list of
                masks that should be subtracted from it.
        """

        def _flatten(
            tree: dict,
            _subtree: dict = None,
        ) -> dict:
            """
            An internal function that performs the actual flattening.

            Args:
                tree (dict):
                    The tree to flatten.

                _subtree (dict, optional):
                    A tree used for flattening the category tree recursively.
                    Internal parameter only.
                    Defaults to None.


            Returns:
                dict:
                    The flattened dictionary
            """
            if _subtree is None:
                _subtree = {}

            for k, v in tree.items():
                if isinstance(v, dict):
                    # Dictionary
                    _subtree[k] = list(v.keys())
                    _flatten(v, _subtree)

                else:
                    # String or None
                    _subtree[k] = []
                    if v is not None:
                        _subtree[v] = []
                        _subtree[k].append(v)

            return _subtree

        return _flatten(categories)

    def remove_overlaps(
        self,
        segmentations: dict,
        categories: dict,
    ):
        """
        Remove overap between masks based on the specified categories.

        Args:
            segmentations (dict):
                Segmentations produced by a segmentation model
                (this is usually implemented in a derived class).

            categories (dict):
                Hierarchical category specification
                (cf. `BaseSegmenter.flatten_categories()`).
        """
        for segmentation in segmentations:
            masks = segmentation["categories"]

            for category, instances in masks.items():

                if dependencies := categories.get(category):

                    for dependency in dependencies:
                        if dependency in categories:
                            segmentations[category][categories[dependency]] = False
