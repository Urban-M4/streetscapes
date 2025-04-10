# --------------------------------------
from pathlib import Path

# --------------------------------------
from PIL import Image

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from streetscapes.models import ModelType
from streetscapes.models import ModelBase
from streetscapes.streetview.instance import SVInstance
from streetscapes.streetview.segmentation import SVSegmentation


class SVImage:
    def __init__(
        self,
        path: Path,
        segmentations: dict[ModelType, SVSegmentation] | None = None,
    ):
        """
        A convenience wrapper around an individual image.
        The wrapper is source-agnostic, meaning that many different
        image sources can be combined at a higher level.

        Args:
            path:
                Path to the image file.

            segmentations:
                A dictionary of segmentations per segmentation model.
                Defaults to None.
        """

        self.path = path
        self.image = np.asarray(Image.open(self.path))
        self.segmentations = {} if segmentations is None else segmentations

    @property
    def tag(self) -> str:
        """
        Return a tag that identifies the image.
        """
        return self.path.stem

    def segmentation(
        self,
        model: ModelType,
    ) -> SVSegmentation:
        """
        Return an SVSegmentation object for a given model.

        Args:
            model:
                The model to use for segmentation.

        Returns:
            The SVSegmentation object.
        """

        # Try to use the cached version
        segmentation = self.segmentations.get(model)

        if segmentation is None:
            # Try to load a saved version
            segmentation = SVSegmentation.from_saved(self.tag, self.path)

        if segmentation is None:

            _model = ModelBase.load_model(model)
            (image_map, masks, instances) = _model.segment_images(self.image)
            self.segmentations[model] = SVSegmentation(image_map[self.id])
            segmentation = self.segmentations[model]

        return segmentation

    def show(self):
        """
        Show the image
        """

        plt.figure()
        plt.imshow(self.image)

    def get_instances(
        self,
        model: ModelType,
        label: str,
    ) -> list[SVInstance]:
        """
        Extract a list of instances.

        Args:
            model:
                The model to get instances for.

            label:
                Label for the requested instances.

        Returns:
            A list of instances.
        """

        mask = self.segmentation(model).get_instances(label=label)

        # TODO: return as Instance object, but currently that doesn't support RGB(A) images
        return np.ma.masked_array(self.image_array, mask=mask)

    def show_instances(self, label: str):

        mask = self.segmentation.get_instances(label=label)

        rgba_image = np.array(self.image.convert("RGBA"))
        rgba_image[..., 3] = np.where(mask.mask.mask, 0, 255)
        plt.imshow(rgba_image[..., :])
