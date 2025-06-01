# --------------------------------------
from pathlib import Path

# --------------------------------------
from PIL import Image

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from streetscapes.streetview.instance import SVInstance
from streetscapes.streetview.segmentation import SVSegmentation


class SVImage:
    """TODO: Add docstrings"""

    def __init__(
        self,
        path: Path,
        segmentations: list[SVSegmentation] | None = None,
    ):
        """
        A convenience wrapper around an individual image.
        The wrapper is source-agnostic, meaning that many different
        image sources can be combined at a higher level.

        Args:
            path:
                Path to the image file.

            segmentations:
                A list of segmentations.
                Defaults to None.
        """

        self.path = path
        self.image = np.asarray(Image.open(self.path))

    @property
    def iid(self) -> str:
        """Return the ID of the image."""
        return self.path.stem

    def segmentations(self) -> list[SVSegmentation]:
        """
        Return a dictionary of models mapped to SVSegmentation objects.

        Returns:
            A dictionary of SVSegmentation objects with the model as key.
        """

        segmentations = {}

        # Search the segmentation directory
        seg_dir = self.path / "segmentations"
        if not seg_dir.exists():
            return segmentations

        for model_subdir in seg_dir.glob("*"):
            if model_subdir.is_dir():
                seg_file = model_subdir / self.path.with_suffix(".npz").name
                if seg_file.is_file():
                    segmentations[model_subdir.name] = SVSegmentation(seg_file)

        return segmentations

    def show(self):
        """
        Show the image
        """

        plt.figure()
        plt.imshow(self.image)

    def get_instances(
        self,
        model: str,
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
