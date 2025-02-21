# --------------------------------------
from pathlib import Path

# --------------------------------------
import PIL
from PIL import Image
import PIL.ImageFile

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------
from copy import deepcopy

# --------------------------------------
import json

# --------------------------------------
import ibis

# --------------------------------------
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
from matplotlib.colors import hex2color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------------------
import streetscapes as scs
from streetscapes import conf
from streetscapes.conf import logger

ImagePath = Path | str | list[Path | str]


class BaseSegmenter:

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

        # Mapping of label ID to label
        self.id_to_label = {}
        self.label_to_id = {}

    def _load(self, *args, **kwargs):
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

    def load_metadata(
        self,
        path: str | Path,
    ) -> tuple[np.ndarray, ak.Array]:
        """
        Load metadata from a Parquet file.

        Args:
            path (str | Path):
                The path to the Parquet file.

        Returns:
            tuple[np.ndarray, ak.Array]:
                A tuple containing:
                    1. A segmentation mask.
                    2. An Awkward array containing metadata about
                        the image and the segmented instances.
        """

        # Load the data into an awkward array
        arr = ak.to_list(ak.from_parquet(path))

        # Load the segmentations
        segmentations = []
        for item in arr:
            segmentations.append(
                {
                    "orig_id": item["orig_id"],
                    "instances": {
                        int(k): v for k, v in zip(item["instance"], item["label"])
                    },
                    "mask": np.array(item.pop("mask"), dtype=np.uint32),
                }
            )

        metadata = ak.Array(arr)

        return (segmentations, metadata)

    def save_metadata(
        self,
        segmentations: list[dict],
        metadata: list[dict],
        path: Path | str,
    ) -> Path:
        """
        Save image metadata to a Parquet file.

        Args:

            segmentations (list[dict]):
                A list of image segmentations.

            metadata (list[dict]):
                A list of metadata entries.

            path (Path | str):
                Path to the output file.

        Returns:
            Path:
                The path to the saved file.
        """

        path = Path(path)
        if path.is_dir():
            raise ValueError("Please provide a valid file path.")

        # Create the directory if it doesn't exist
        scs.mkdir(path.parent)

        # Save only those segmentations that have associated
        # metadata entries
        orig_ids = {meta["orig_id"]: meta for meta in ak.to_list(metadata)}

        filtered = []
        for seg in segmentations:

            if seg["orig_id"] not in orig_ids:
                continue

            merged = {"mask": seg["mask"].tolist()}
            merged.update(orig_ids[seg["orig_id"]])
            filtered.append(merged)

        # Convert the stats into a JSON object and
        # then into an Awkward array.
        arr = ak.from_json(json.dumps(filtered))

        # Save the array to a Parquet file.
        ak.to_parquet(arr, path)

        return path

    def load_images(
        self,
        images: ImagePath,
    ) -> dict[Path, np.ndarray]:
        """
        A list of images or paths to image files.

        Args:
            images (ImagePath):
                A path or a list of paths to image files.

        Returns:
            tuple[list[Path], list[np.ndarray]]:
                A tuple containing:
                    1. The paths to the images.
                    2. The images as NumPy arrays.
        """

        if isinstance(images, (str, Path)):
            # Ensure that we have an iterable.
            images = [images]

        image_paths = [Path(img) for img in images]
        images = [np.array(Image.open(image_path)) for image_path in image_paths]

        return (image_paths, images)

    def extract_metadata(
        self,
        images: list[np.ndarray],
        segmentations: list[dict],
    ) -> ak.Array:
        """
        Compute colour statistics and other metadata
        for a list of segmented images.

        Args:

            images (list[np.ndarray]):
                A list of images as NumPy arrays.

            segmentations (list[dict]):
                A list of dictionaries containing instance-level segmentation details.
                Each instance is denoted with its ID (= the keys in the `instances` dictionary).

        Returns:
            list[dict]:
                A dictionary of metadata.
        """

        logger.info("Extracting metadata...")

        # Extract some extra image attributes.
        # First, filter the full database to find the right entries.
        full_dataset = ibis.read_parquet(conf.PARQUET_DIR / "streetscapes.parquet")
        extra_data = (
            full_dataset.select("orig_id", "lat", "lon")
            .filter(
                full_dataset["orig_id"].isin(
                    set([seg["orig_id"] for seg in segmentations])
                )
            )
            .to_pandas()
        )

        results = []

        # Loop over the segmented images and compute
        # some statistics for each instance.
        for image, segmentation in zip(images, segmentations):

            # Unpack the segmentation dictionary
            orig_id = segmentation["orig_id"]
            instances = segmentation["instances"]
            mask = segmentation["mask"]

            # Filter based on the availability of (lat, lon)
            row = extra_data[extra_data["orig_id"] == orig_id].values
            if len(row) == 0:
                # Skip the image if there are no coordinates
                continue

            # Create a metadata dictionary for this segmentation
            metadata = {
                "orig_id": orig_id,
                "instance": [],
                "label": [],
                "R": [],
                "G": [],
                "B": [],
                "H": [],
                "S": [],
                "V": [],
                "area": [],
                "lat": row[0][1],
                "lon": row[0][2],
            }

            # Convert the image colour space to floating-point RGB and HSV
            rgb_image = ski.exposure.rescale_intensity(image, out_range=(0, 1))
            hsv_image = ski.color.convert_colorspace(rgb_image, "RGB", "HSV")
            with tqdm(total=len(instances)) as pbar:
                for inst_id, label in instances.items():
                    metadata["instance"].append(inst_id)
                    metadata["label"].append(label)

                    # Extract the patches corresponding to the mask
                    inst_mask = mask == inst_id
                    rgb_patch = rgb_image[inst_mask]
                    hsv_patch = hsv_image[inst_mask]

                    # Extract the R, G, B, H, S and V channels
                    # to compute the corresponding statistics.
                    patches = {
                        "R": rgb_patch[..., 0],
                        "G": rgb_patch[..., 1],
                        "B": rgb_patch[..., 2],
                        "H": hsv_patch[..., 0],
                        "S": hsv_patch[..., 1],
                        "V": hsv_patch[..., 2],
                    }

                    for k, v in patches.items():
                        metadata[k].append(
                            {
                                "median": np.median(v),
                                "mode": scipy.stats.mode(v)[0],
                                "mean": np.mean(v),
                                "sd": np.std(v),
                            }
                        )

                    metadata["area"].append(
                        np.count_nonzero(mask) / np.prod(rgb_image.shape[:2])
                    )

                    pbar.update()

            results.append(metadata)

        return ak.Array(results)

    def visualise_segmentation(
        self,
        image: np.ndarray,
        segmentation: dict,
        labels: list[str] = None,
        opacity: float = 0.5,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Visualise the instances of different objects in an image.

        Args:
            image (np.ndarray):
                Image being segmented.

            segmentation (dict):
                A dictionary contianing an instance segmentation mask
                and a mapping of instance IDs to labels.

            labels (list[str]):
                Labels for the instance categories that should be plotted.

            opacity (float, optional):
                Opacity to use for the segmentation overlay.
                Defaults to 0.5.

            title (str | None, optional):
                The figure title.
                Defaults to None.

            figsize (tuple[int, int], optional):
                Figure size. Defaults to (16, 6).

        Returns:
            tuple[plt.Figure, plt.Axes]:
                A tuple containing:
                    - A Figure object.
                    - An Axes object that allows further annotations to be added.
        """

        # Prepare the greyscale version of the image for plotting instances.
        greyscale = scs.as_rgb(image, greyscale=True)

        # Create a figure
        (fig, axes) = plt.subplots(1, 2, figsize=figsize)

        # Prepare the colour dictionary and the layers
        # necessary for plotting the category patches.
        colourmap = scs.make_colourmap(labels)

        # Label handles for the plot legend.
        handles = {}

        # Get the mask and the instances
        mask = segmentation["mask"]
        instances = segmentation['instances']

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

    def segment_from_dataset(
        self,
        dataset: Path | str,
        labels: dict,
        image_dir: Path | str = None,
        sample: int = None,
        save: bool = False,
    ) -> tuple[dict, Path | None]:
        """
        Segment images specified in a Parquet data (sub)set.

        Args:
            dataset (Path | str):
                The input dataset (a Parquet file).

            labels (dict):
                A flattened set of labels to look for,
                with optional subsets of labels that should be
                checked in order to eliminate overlaps.
                Cf. `BaseSegmenter._flatten_labels()`

            image_dir (Path | str):
                The directory where downloaded images are located.
                Defaults to None.

            sample (int, optional):
                The size of the sample (used for testing purposes).

            save (bool, optional):
                Save the computed segmentation and statistics.
                Defaults to False.

        Returns:
            list[dict]:
                The statistics for all the segmented images.
        """

        # Load the Parquet dataset.
        # ==================================================
        dataset = ibis.read_parquet(dataset).to_pandas()

        # Directory containing the downloaded images
        # ==================================================
        if image_dir is None:
            image_dir = conf.IMAGE_DIR

        paths = [image_dir / f"{file_name}.jpeg" for file_name in dataset["orig_id"]]

        if sample is not None:
            paths = list(Path(n) for n in np.random.choice(paths, sample))

        # Segment the images
        # ==================================================
        segmentations = self.segment(paths, labels)

        return segmentations

    def _flatten_labels(
        self,
        labels: dict,
    ) -> dict:
        """
        Flatten a nested dictionary of labels.

        Useful for defining masks in terms of more general labels
        that can be subtracted.
        For instance, building facades can include windows and doors,
        which means that we can define a category dictionary as follows:

        labels = {
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
                The labels as a tree (dictionary of dictionaries).

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

        return _flatten(labels)
