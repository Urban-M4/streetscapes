# --------------------------------------
from __future__ import annotations

# --------------------------------------
from abc import ABC
from abc import abstractmethod

# --------------------------------------
from pathlib import Path

# --------------------------------------
import PIL
from PIL import Image
import PIL.ImageFile

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------
import enum

# --------------------------------------
import json

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import numpy as np

# --------------------------------------
import scipy

# --------------------------------------
import skimage as ski

# --------------------------------------
import typing as tp

# --------------------------------------
import streetscapes as scs
from streetscapes import utils
from streetscapes.utils import logger
from streetscapes.utils import CIEnum
from streetscapes.utils.enums import Attr
from streetscapes.utils.enums import Stat

ImagePath = Path | str | list[Path | str]


class ModelType(CIEnum):
    """
    An enum listing supported models.
    """

    MaskFormer = enum.auto()
    DinoSAM = enum.auto()


class ModelBase(ABC):

    models = {}

    @staticmethod
    @abstractmethod
    def get_model_type() -> ModelType:
        """
        Get the enum corresponding to this model.
        """
        pass

    @staticmethod
    def _get_derived(
        parent: tp.Any | None = None,
        models: dict | None = None,
    ) -> dict[ModelType, tp.Any]:

        if models is None:
            models = {}

        if parent is None:
            parent = ModelBase

        for model_cls in parent.__subclasses__():
            try:

                mtype = model_cls.get_model_type()
                if mtype is not None:
                    models[mtype] = model_cls
            except AttributeError as e:
                pass

            # Recurse
            ModelBase._get_derived(model_cls, models)

        return models

    @staticmethod
    def load_model(model_type: ModelType, *args, **kwargs) -> ModelBase:
        """
        Load a model.

        Args:
            model:
                An enum specifying the model to load.

        Returns:
            The loaded model.
        """
        if model_type not in ModelBase.models:
            model_cls = ModelBase._get_derived().get(model_type)
            if model_cls is None:
                return

            ModelBase.models[model_type] = model_cls(*args, **kwargs)

        return ModelBase.models[model_type]

    def __init__(
        self,
        device: str = None,
    ):
        """
        A model serving as the base for all segmentation models.

        Args:
            device:
                Device to use for processign. Defaults to None.
        """
        import torch

        # Name
        # ==================================================
        self.name = self.get_model_type().name
        self.env_prefix = utils.camel2snake(self.name).upper()

        # Set up the device
        # ==================================================
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Mapping of label ID to label
        self.id_to_label = {}
        self.label_to_id = {}

    @abstractmethod
    def _from_pretrained(self, *args, **kwargs):
        """
        Load a pretrained model.

        NOTE
        This method that should be overriden in derived classes.
        """
        pass

    @abstractmethod
    def segment_images(self, *args, **kwargs):
        """
        Segments a list of images and looks for the requested labels.

        NOTE
        This method that should be overriden in derived classes.
        """
        pass

    def load_stats(
        self,
        paths: str | Path | list[str | Path],
    ) -> dict[int, dict]:
        """
        Load metadata from a Parquet file.

        Args:
            paths (str | Path | list[str | Path]):
                The paths to the Parquet file containing image statistics.

        Returns:
            dict[int, dict]:
                A dictionary mapping origin IDs to dictionaries
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
            for attr, attr_data in loaded:
                try:
                    attr = Attr[attr.capitalize()]
                except:
                    attr = attr.lower()
                match attr:
                    case "instance" | "label" | Attr.Area | Attr.Coords:
                        stats[orig_id][attr] = attr_data
                    case _:
                        # Statistics
                        stats[orig_id][attr] = {}
                        for stat, stat_data in attr_data.items():
                            stat = Stat[stat.capitalize()]
                            stats[orig_id][attr][stat] = stat_data

        return stats

    def save_stats(
        self,
        stats: dict,
        path: Path | None = None,
    ) -> list[Path]:
        """
        Save image metadata to a Parquet file.

        Args:

            stats:
                A list of metadata entries.

            path:
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
    ) -> dict[int, dict]:
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
            dict[int, dict]:
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
                                np.count_nonzero(inst_mask)
                                / np.prod(rgb_image.shape[:2])
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


    def load_images(
        self,
        images: ImagePath,
    ) -> tuple[list[Path], list[np.ndarray]]:
        """
        A list of images or paths to image files.

        Args:
            images:
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
            labels:
                The labels as a tree (dictionary of dictionaries).

        Returns:
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
