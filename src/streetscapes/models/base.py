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
import itertools

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
from streetscapes.enums import Stat
from streetscapes.enums import Attr
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

    def load_stats(
        self,
        paths: str | Path | list[str | Path],
    ) -> ak.Array:
        """
        Load metadata from a Parquet file.

        Args:
            paths (str | Path | list[str | Path]):
                The paths to the Parquet file containing image statistics.

        Returns:
            dict[int, ak.Array]:
                A dictionary mapping origin IDs to Awkward arrays
                containing statistics about the segmented images.
        """

        # Ensure that we have a list of path objects
        if isinstance(paths, str):
            paths = [paths]
        paths = [Path(p) for p in paths]

        stats = {}

        # Load the statistics
        for path in paths:
            orig_id = int(str(path.name).removesuffix("".join(path.suffixes)))
            stats[orig_id] = {}
            loaded = json.loads(ak.to_json(ak.from_parquet(path))).items()
            for key, data in loaded:
                print(f"==[ key: {key}")
                match key:
                    case "instance" | "label":
                        stats[orig_id][key] = data
                    case "area":
                        stats[orig_id][Attr[key.capitalize()]] = data
                    case _:
                        print(f"==[ {key}: {data}")
                        if key in Attr.JSONColour:
                            # Statistics
                            stats[orig_id][Attr[key.capitalize()]] = {}
                            for s, v in data.items():
                                stats[orig_id][Attr[key.capitalize()]][
                                    Stat[s.capitalize()]
                                ] = v

        return stats

    def save_stats(
        self,
        stats: dict,
        path: Path | None = None,
    ) -> list[Path]:
        """
        Save image metadata to a Parquet file.

        Args:

            stats (list[dict]):
                A list of metadata entries.

            path (Path | None, optional):
                A directory where the stats should be saved.
                Defaults to None.

        Returns:
            list[Path]:
                A list of paths to the saved files.
        """

        if path is None:
            path = conf.PARQUET_DIR

        path = scs.mkdir(path)

        files = []

        for orig_id, stat in stats.items():

            # File path
            fpath = path / f"{orig_id}.stat.parquet"

            # Convert the stats into a JSON object and
            # then into an Awkward array.
            arr = ak.from_json(json.dumps(stat))

            # Save the array to a Parquet file.
            ak.to_parquet(arr, fpath)

            files.append(fpath)

        return files

    def extract_stats(
        self,
        images: dict[int, np.ndarray],
        masks: dict[int, np.ndarray],
        instances: dict[int, dict],
        attrs: set[Attr] | None = None,
        stats: set[Stat] | None = None,
    ) -> ak.Array:
        """
        Compute colour statistics and other metadata
        for a list of segmented images.

        Args:

            images (dict[int, np.ndarray]):
                A dictionary of images as NumPy arrays.

            masks (dict[int, np.ndarray]):
                A dictionary of masks as NumPy arrays.

            instances (dict[int, dict]):
                A dictionary of instances containing instance-level segmentation details.
                Each instance is denoted with its ID (= the keys in the `instances` dictionary).

            attrs (set[Attr] | None, optional):
                A set of Attr items representing image attributes.
                Defaults to None.

            stats (set[Stat] | None, optional):
                A set of Stats items representing statistics to be extracted
                from the segmentations.
                Defaults to None.

        Returns:
            list[dict]:
                A dictionary of metadata.
        """

        logger.info("Extracting metadata...")

        # Statistics about instances in the images
        image_stats = {}

        # Ensure that the attrs and stats are sets
        attrs = set(attrs) if attrs is not None else set()
        stats = set(stats) if stats is not None else set()

        rgb = len(attrs.intersection(Attr.RGB)) > 0
        hsv = len(attrs.intersection(Attr.HSV)) > 0

        # Loop over the segmented images and compute
        # some statistics for each instance.
        for orig_id, image in images.items():

            # Create a new dictionary to hold the results
            computed = image_stats.setdefault(
                orig_id,
                {
                    "instance": [],
                    "label": [],
                },
            )

            # Convert the image colour space to floating-point RGB and HSV
            if rgb or hsv:
                rgb_image = ski.exposure.rescale_intensity(image, out_range=(0, 1))
            if hsv:
                hsv_image = ski.color.convert_colorspace(rgb_image, "RGB", "HSV")

            # Ensure that the mask and the instances for this image are available
            mask = masks.get(orig_id)
            if mask is None:
                continue

            image_instances = instances.get(orig_id)
            if image_instances is None:
                continue

            with tqdm(total=len(image_instances)) as pbar:
                for inst_id, label in image_instances.items():
                    computed["instance"].append(inst_id)
                    computed["label"].append(label)

                    # Extract the patches corresponding to the mask
                    patches = {}
                    inst_mask = mask == inst_id
                    if rgb:
                        rgb_patch = rgb_image[inst_mask]
                        patches.update(
                            {
                                Attr.R: rgb_patch[..., 0],
                                Attr.G: rgb_patch[..., 1],
                                Attr.B: rgb_patch[..., 2],
                            }
                        )
                    if hsv:
                        hsv_patch = hsv_image[inst_mask]
                        patches.update(
                            {
                                Attr.H: hsv_patch[..., 0],
                                Attr.S: hsv_patch[..., 1],
                                Attr.V: hsv_patch[..., 2],
                            }
                        )

                    # Extract the statistics for the requested attributes.
                    for attr in attrs:
                        if attr == Attr.Area:
                            computed.setdefault(attr, []).append(
                                np.count_nonzero(mask) / np.prod(rgb_image.shape[:2])
                            )
                        else:
                            computed.setdefault(attr, {stat: [] for stat in stats})
                            for stat in stats:
                                match stat:
                                    case Stat.Median:
                                        value = np.nan_to_num(
                                            np.median(patches[attr]), nan=0.0
                                        )
                                    case Stat.Mode:
                                        value = np.nan_to_num(
                                            scipy.stats.mode(patches[attr])[0], nan=0.0
                                        )
                                    case Stat.Mean:
                                        value = np.nan_to_num(
                                            np.mean(patches[attr]), nan=0.0
                                        )
                                    case Stat.SD:
                                        value = np.nan_to_num(
                                            np.std(patches[attr]), nan=0.0
                                        )
                                    case _:
                                        value = None

                                if value is not None:
                                    computed[attr][stat].append(value)

                    pbar.update()

        return image_stats

    def save_masks(
        self,
        masks: dict,
        path: Path | None = None,
    ) -> list[Path]:
        """
        Save image segmentation masks to NPZ files.

        Args:

            masks (list[dict]):
                A dictionary containing image segmentation masks.

            path (Path | None, optional):
                A directory where the masks should be saved.
                Defaults to None.

        Returns:
            list[Path]:
                A list of paths to the saved files.
        """

        if path is None:
            path = conf.IMAGE_DIR

        path = scs.mkdir(path)

        files = []

        for orig_id, mask in masks.items():

            # File path
            fpath = path / f"{orig_id}.npz"

            # Save the mask as a compressed NumPy array.
            np.savez_compressed(fpath, mask)

            files.append(fpath)

        return files

    def load_masks(
        self,
        paths: str | Path | list[str | Path],
    ) -> dict[int, np.ndarray]:
        """
        Load segmentation masks from NumPy archives.

        Args:
            paths (str | Path | list[str | Path]):
                Path(s) to the archives.

        Returns:
            dict[int, np.ndarray]:
                A dictionary mapping origin IDs to NumPy arrays
                containing the segmentation masks.
        """

        # Ensure that we have a list of path objects
        if isinstance(paths, str):
            paths = [paths]
        paths = [Path(p) for p in paths]

        masks = {}

        for path in paths:
            masks[int(path.stem)] = np.load(path, allow_pickle=False)["arr_0"]

        return masks

    def load_images(
        self,
        images: ImagePath,
    ) -> tuple[list[Path], list[np.ndarray]]:
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

    def load_image(
        self,
        path: Path | str,
    ) -> tuple[list[Path], list[np.ndarray]]:
        """
        A list of images or paths to image files.

        Args:
            path (Path | str):
                A path to an image.

        Returns:
            tuple[int, np.ndarray]:
                A tuple containing:
                    1. The ID of the image.
                    2. The images as a NumPy array.
        """

        return int(path.stem), np.array(Image.open(path))

    def visualise_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instances: dict,
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

            mask (np.ndarray):
                The segmentation mask.

            instances (dict):
                A dictionary of instances and their labels.

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
        dataset: Path | str | ibis.Table | pd.DataFrame,
        labels: dict,
        sample: int | None = None,
        batch_size: int = 10,
        ensure: list[str] | None = None,
        attrs: set[Attr] | None = None,
        stats: set[Stat] | None = None,
    ) -> tuple[dict, Path | None]:
        """
        Segment images specified in a Parquet data (sub)set.

        Args:
            dataset (Path | str | ibis.Table | pd.DataFrame):
                The input dataset (a Parquet file, an Ibis table or a Pandas dataframe).

            labels (dict):
                A flattened set of labels to look for,
                with optional subsets of labels that should be
                checked in order to eliminate overlaps.
                Cf. `BaseSegmenter._flatten_labels()`

            sample (int, optional):
                The size of the sample (used for testing purposes).
                Defaults to None.

            batch_size (int, optional):
                Process the images in batches of this size.
                Defaults to 10.

            ensure (list[str] | None, optional):
                A list of columns whose values must be set (not null or empty).
                Defaults to None.

            attrs (set[Attr] | None, optional):
                Attributes to use for computing the segmentation.
                Defaults to None.

            stats (set[Stat] | None, optional):
                Statistics to compute for each attribute.
                Defaults to None.

        Returns:
            list[dict]:
                The statistics for all the segmented images.
        """

        # Load the Parquet dataset.
        # ==================================================
        if isinstance(dataset, (str, Path)):
            # Read the valuate to a Pandas dataframe
            dataset = ibis.read_parquet(dataset)

        if isinstance(dataset, ibis.Table):
            # Evaluate to a Pandas dataframe
            dataset = dataset.to_pandas()

        # Drop rows with missing values
        if ensure is not None:
            dataset = dataset.dropna(subset=ensure)

        # Build a list of paths from the file names.
        paths = [
            conf.IMAGE_DIR / f"{file_name}.jpeg" for file_name in dataset["orig_id"]
        ]

        # Extract a sample of a certain size.
        if sample is not None:
            paths = list(Path(n) for n in np.random.choice(paths, sample))

        # Ensure that the attrs and stats are sets
        attrs = set(attrs) if attrs is not None else set()
        stats = set(stats) if stats is not None else set()

        # Segment the images and save the results
        # ==================================================
        total = len(paths) // batch_size
        if total * batch_size != len(paths):
            total += 1

        # Lists of paths to the saved segmentations and stats
        image_paths = []
        mask_paths = []
        stat_paths = []

        # Segment the images and extract the metadata
        for batch in tqdm(itertools.batched(paths, batch_size), total=total):
            images, masks, instances = self.segment(batch, labels)

            image_paths.extend(batch)

            if stats is not None:
                image_stats = self.extract_stats(images, masks, instances, attrs, stats)

            mask_paths.extend(self.save_masks(masks))
            stat_paths.extend(self.save_stats(image_stats))

        return image_paths, mask_paths, stat_paths

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
