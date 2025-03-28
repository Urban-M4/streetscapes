# --------------------------------------
from pathlib import Path

# --------------------------------------
from PIL import Image

# --------------------------------------
import numpy as np

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from streetscapes import models
from streetscapes.streetview.segmentation import SVSegmentation

class SVImage:
    def __init__(self, path: Path):
        self.path = path
        self.image = Image.open(self.path)
        self.image_array = np.asarray(self.image)

        self.models = {}

    def id(self) -> int:
        return int(self.path.stem)

    def mask_path(self) -> Path:

        # TODO: perhaps add the model that was used as subdir?
        return self.path.with_suffix("npz")

    def segment(self, model) -> models.BaseSegmenter:
        if model not in self.models:
            self.models[model] = models.load_model(model)

        self.models[model].segment(self.path)

    def plot(self):
        plt.figure()
        plt.imshow(self.image)

    @property
    def segmentation(self) -> SVSegmentation:
        return SVSegmentation.from_saved(self.path)

    def get_instances(self, label: str):
        mask = self.segmentation.get_instances(label=label)
        # TODO: return as Instance object, but currently that doesn't support RGB(A) images
        return np.ma.masked_array(self.image_array, mask=mask)

    def plot_instances(self, label: str):
        mask = self.segmentation.get_instances(label=label)

        rgba_image = np.array(self.image.convert("RGBA"))
        rgba_image[..., 3] = np.where(mask.mask.mask, 0, 255)
        plt.imshow(rgba_image[..., :])
