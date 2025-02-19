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

ImageOrPath = (
    Path
    | str
    | np.ndarray
    | Image.Image
    | list[Path | str | np.ndarray | Image.Image]
    | tuple[Path | str | np.ndarray | Image.Image]
)


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
    ) -> dict:
        """
        Load metadata from a Parquet file.

        Args:
            path (str | Path):
                The path to the Parquet file.

        Returns:
            dict:
                A dictionary of image metadata.
        """

        # Load the data
        arr = ak.from_parquet(path)

        # Convert the masks and the outlines into NumPy arrays
        metadata = json.loads(ak.to_json(arr))
        metadata["masks"] = np.array(metadata["masks"], dtype=np.uint32)
        metadata["outlines"] = np.array(metadata["outlines"], dtype=np.uint32)
        metadata['stats'] = ak.Array[metadata['stats']]

        return metadata

    def save_metadata(
        self,
        stats: dict,
        path: Path | str,
    ) -> Path:
        """
        Save image metadata as an Awkward array to a Parquet file.

        Args:

            stats (dict):
                A dictionary of statistics.

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

        # Convert the masks and the outlines into lists.
        _stats = deepcopy(stats)
        try:
            _stats["masks"] = stats["masks"].tolist()
            _stats["outlines"] = stats["outlines"].tolist()
        except AttributeError as e:
            pass

        # Convert the stats into a JSON object and
        # then into an Awkward array.
        arr = ak.from_json(json.dumps(_stats))

        # Save the array to a Parquet file.
        ak.to_parquet(arr, path)

        return path

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

        if isinstance(images, (str, Path, np.ndarray, Image.Image)):
            # A single image
            images = [images]

        if isinstance(images, (list, tuple)):
            images = {f"image_{n}": image for n, image in enumerate(images)}

        result = {}
        for image_name, image in images.items():
            if isinstance(image, (str, Path)):
                image_name = image.name
                image = Image.open(image)

            result[image_name] = np.array(image)

        return result

    def compute_stats(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        instances: dict[int, str],
    ) -> ak.Array:
        """
        Compute the statistics for all labels.

        Args:
            image (np.ndarray):
                The original image.
                Needed for extracting colour information for each instance.

            masks (np.ndarray):
                A segmentation mask containing pixel-level instance information.
                Each instance is denoted with its ID (= the keys in the `instances` dictionary).

            instances (dict[int, str]):
                A mapping from instance ID (int) to its label.
                This is used to extract the individual instance masks from the image
                and computing the statistics for the corresponding patch.

        Returns:
            dict:
                A dictionary of statistics.
        """

        # Loop over the segmented images and compute
        # some statistics for each patch.
        logger.info("Computing statistics...")

        stats = {
            "instance": [],
            "label": [],
            "R": [],
            "G": [],
            "B": [],
            "H": [],
            "S": [],
            "V": [],
            "area": [],
        }

        # Convert the image colour space
        rgb_image = ski.exposure.rescale_intensity(image, out_range=(0, 1))
        hsv_image = ski.color.convert_colorspace(rgb_image, "RGB", "HSV")
        with tqdm(total=len(instances)) as pbar:
            for inst_id, label in instances.items():
                stats["instance"].append(inst_id)
                stats["label"].append(label)

                # Extract the patches corresponding to the mask
                mask = masks == inst_id
                rgb_patch = rgb_image[mask]
                hsv_patch = hsv_image[mask]

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
                    stats[k].append(
                        {
                            "median": np.median(v),
                            "mode": scipy.stats.mode(v)[0],
                            "mean": np.mean(v),
                            "sd": np.std(v),
                        }
                    )

                stats["area"].append(
                    np.count_nonzero(mask) / np.prod(rgb_image.shape[:2])
                )

                pbar.update()

        return stats

    def visualise_segmentation(
        self,
        image: np.ndarray,
        metadata: dict,
        labels: list[str] = None,
        opacity: float = 0.5,
        outline: str = "#ffff00",
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Visualise the instances of different objects in an image.

        Args:
            image (np.ndarray):
                Image being segmented.

            metadata (dict):
                A dictionary contianing metadata, including the
                instance segmentation and outline maps.

            labels (list[str]):
                Labels for the instance categories that should be plotted.

            opacity (float, optional):
                Opacity to use for the segmentation overlay.
                Defaults to 0.5.

            outline (str, optional):
                Outline colour for instances.
                Defaults to "#ffff00".

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

        # Convert the outline colour into a NumPy array
        if outline is not None:
            outline = (255 * np.array(hex2color(outline))).astype(np.ubyte)

        # Get the masks and the outlines
        masks = metadata["masks"]
        outlines = metadata["outlines"]
        stats = metadata["stats"]
        instances = {
            inst_id: inst_label
            for inst_id, inst_label in zip(
                stats["instance"],
                stats["label"],
            )
        }

        # Loop over the segmentation list
        outline_layer = np.zeros(greyscale.shape[:2])
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
            mask = masks == instance_id
            if not np.any(mask):
                continue

            greyscale[mask] = (
                (1 - opacity) * greyscale[mask] + 255 * opacity * colourmap[label]
            ).astype(np.ubyte)

            if outline is not None:
                # Extract the outline
                outline_layer[outlines == instance_id] = 1

        # Plot the original image and the segmented one.
        # If any of the requested categories exist in the
        # image, they will be overlaid as coloured patches
        # with the given opacity over the original image.
        # ==================================================
        if outline is not None:
            greyscale[outline_layer > 0] = outline
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(greyscale)
        axes[1].axis("off")
        axes[1].legend(
            handles=handles.values(), loc="upper left", bbox_to_anchor=(1, 1)
        )
        if title is not None:
            fig.suptitle(f"Segmentation for {title}", fontsize=16)

        return (fig, axes)

    def segment_from_dataset(
        self,
        path: Path | str,
        labels: dict,
        image_dir: Path | str = None,
        sample: int = None,
        save: bool = True,
    ) -> tuple[dict, Path | None]:
        """
        Segment images specified in a CSV data (sub)set.

        Args:
            path (Path | str):
                The input dataset (a CSV file).

            labels (dict):
                A flattened set of labels to look for,
                with optional subsets of labels that should be
                checked in order to eliminate overlaps.
                Cf. `BaseSegmenter._flatten_labels()`

            image_dir (Path | str):
                The directory where downloaded images are located. Defaults to None.

            sample (int, optional):
                The size of the sample (used for testing purposes).

            save (bool, optional):
                Save the computed segmentation and statistics.
                Defaults to True.

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

        paths = [image_dir / f"{file_name}.jpeg" for file_name in df["orig_id"]]

        if sample is not None:
            paths = list(Path(n) for n in np.random.choice(paths, sample))

        # Load the images
        image_data = self.load_images(paths)

        # Segment the images
        # ==================================================
        metadata = self.segment(image_data, labels)

        # Save the dataset if a file name is provided
        # ==================================================
        if save:
            for path, entry in zip(paths, metadata):
                self.save_metadata(entry, image_dir / path.with_suffix(".parquet"))

        return (paths, metadata)

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
