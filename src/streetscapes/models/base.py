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
import numpy as np

# --------------------------------------
import typing as tp

# --------------------------------------
from streetscapes import utils
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
                tree:
                    The tree to flatten.

                _subtree:
                    A tree used for flattening the category tree recursively.
                    Internal parameter only.
                    Defaults to None.


            Returns:
                The flattened dictionary.
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
        path: str | Path | list[str | Path],
    ) -> dict[int, dict]:
        """
        Load metadata from a Parquet file.

        Args:
            paths:
                The paths to the Parquet file containing image statistics.

        Returns:
            A dictionary mapping origin IDs to dictionaries
            containing statistics about the segmented images.
        """

        # Ensure that we have a list of path objects
        if isinstance(path, str):
            path = [path]
        path = [Path(p) for p in path]

        stats = {}

        # Load the statistics
        for path in path:
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
            path:
                A path to an image.

        Returns:
            A tuple containing:
                1. The ID of the image.
                2. The images as a NumPy array.
        """

        return int(path.stem), np.array(Image.open(path))
