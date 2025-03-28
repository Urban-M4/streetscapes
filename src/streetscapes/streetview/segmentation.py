import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

import ibis

from streetscapes.streetview.instance import SVInstance


class SVSegmentation:

    def __init__(
        self,
        mask: np.ndarray,
        labels: dict[int, str] = None,
    ):
        self.mask = mask
        self.labels = labels or {}

    @classmethod
    def from_saved(cls, image_path: Path):
        """Load mask and labels from previously saved segmentation."""

        mask_file = image_path.with_suffix(".npz")
        label_file = image_path.with_suffix(".stat.parquet")

        try:
            mask = np.load(mask_file, allow_pickle=False)["arr_0"]
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Mask file not found. Have you segmented the image?"
            ) from exc

        try:
            stats = ibis.read_parquet(label_file).loc[0]
            labels = dict(zip(stats.instance.tolist(), stats.label))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Label file not found. Have you segmented the image with stats calculation?"
            ) from exc

        return cls(mask, labels)

    def visualise(self):
        """Visualise the segmentation mask."""
        plt.figure()
        plt.imshow(self.mask)

    def __repr__(self):
        return f"Segmentation(mask={self.mask}, labels={self.labels})"

    def get_instances(self, label: str):
        """Return an array of instances corresponding to label."""
        index = [k for (k, v) in self.labels.items() if v == label][0]
        instance_mask = np.ma.masked_where(self.mask != index, self.mask)
        return SVInstance(instance_mask)
