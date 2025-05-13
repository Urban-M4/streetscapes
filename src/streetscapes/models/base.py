# --------------------------------------
from __future__ import annotations

# --------------------------------------
import os

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
import itertools

# --------------------------------------
import ibis

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import numpy as np

# --------------------------------------
import awkward as ak

# --------------------------------------
from streetscapes import utils
from streetscapes.streetview import SVSegmentation

PathLike = Path | str | list[Path | str]

# HACK
# PyTorch configuration options.
# This should go into a dedicated configuration module.
# ==================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class ModelBase(ABC):

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
        self.name = self.__class__.__name__.lower()
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
    def _segment_images(self, *args, **kwargs):
        """
        Segments a list of images and looks for the requested labels.

        NOTE
        This method that should be overriden in derived classes.
        """
        pass

    def _save_segmentation(
        self,
        segmentation: dict,
    ):
        """
        Save an image segmentation mask and the corresponding instances.

        Args:
            segmentation:
                A dictionary containing image segmentation information.
        """

        # Masks and instances are saved in separate directories for each model,
        # so it's useful to have the model name handy.
        model_name = self.name.lower()

        # The directory where the image is stored
        image_path = segmentation["image_path"]
        image_dir = image_path.parent

        # Derive the mask and instance directories from the image directory
        segmentation_dir = utils.ensure_dir(image_dir / f"segmentations/{model_name}")

        # Reformat the dictionary values and save.
        # ==================================================
        to_save = {
            "image_name": image_path.name,
            "masks": list(segmentation["masks"].items()),
            "instances": list(segmentation["instances"].items()),
        }

        path = segmentation_dir / image_path.with_suffix(".npz").name

        # ak.to_parquet(to_save, path)
        np.savez_compressed(path, to_save)

        return path

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

    def load_images(
        self,
        images: PathLike,
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

    def segment(
        self,
        paths: PathLike,
        labels: dict,
        batch_size: int = 10,
    ) -> ibis.Table:
        """
        Retrieve the paths of local images from a dataset.

        Args:
            paths:
                Path(s) to image file(s).

            labels:
                A flattened set of labels to look for,
                with optional subsets of labels that should be
                checked in order to eliminate overlaps.
                Cf. `BaseSegmenter._flatten_labels()`

            batch_size:
                Process the images in batches of this size.
                Defaults to 10.

        Returns:
            A table of information about the segmentations.
        """

        # Segment the images and save the results
        # ==================================================
        segmentations = []

        # Make sure that we are processing a list
        single = isinstance(paths, (str, Path))
        if single:
            paths = [paths]

        # Compute the number of batches
        total = len(paths) // batch_size
        if total * batch_size != len(paths):
            total += 1

        # Segment the images and extract the metadata
        pbar = tqdm(total=total, desc=f"Segmenting images...")
        for path_batch in itertools.batched(list(paths), batch_size):
            segmentations.extend(self._segment_images(path_batch, labels))
            pbar.update()

        # Save the images
        pbar.set_description_str("Saving segmentations...")

        # Save the segmentations
        paths = [self._save_segmentation(seg) for seg in segmentations]

        # Create SVSegmentation instances
        segmentations = [SVSegmentation(path) for path in paths]
        pbar.set_description_str("Done")

        return segmentations[0] if single else segmentations
