from pathlib import Path
import numpy as np
import typing as tp
from streetscapes import utils


class SVInstance:
    """Represents a single segmented instance in an image."""

    def __init__(
        self,
        image_path: Path,
        mask: np.ndarray,  # boolean mask shape (H, W)
        label: str,
        iid: int,
    ):
        if mask.dtype != bool:
            raise ValueError("mask must be a boolean numpy array")
        self.image_path = image_path
        self.mask = mask
        self.label = label
        self.iid = iid

    def __repr__(self):
        return f"SVInstance(label={self.label}, iid={self.iid})"

    def masked(
        self,
        channel: int = None,
        hsv: bool = False,
    ) -> np.ndarray:
        """
        Return the pixel values corresponding to this instance's mask.

        Args:
            channel: Specific channel to extract (e.g. 0=R, 1=G, 2=B).
            hsv: If True, convert to HSV before extracting pixels.

        Returns:
            np.ndarray: The masked pixel values (N, C) or (N,) if single channel.
        """
        image = utils.open_image(self.image_path)
        if hsv:
            image = utils.as_hsv(image)

        pixels = image[self.mask]
        return pixels if channel is None else pixels[..., channel]

    def brightness(self) -> np.ndarray:
        """Convenience method: get the brightness channel from HSV space."""
        return self.masked(hsv=True, channel=2)

    def visualise(
        self,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
        channel: int = None,
    ):
        """
        Visualise this instance isolated, with everything else blacked out.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1, figsize=figsize)

        image = utils.open_image(self.image_path)
        canvas = np.zeros_like(image)

        if isinstance(channel, int) and channel < image.shape[-1]:
            canvas[..., channel][self.mask] = image[..., channel][self.mask]
        else:
            canvas[self.mask] = image[self.mask]

        axes.imshow(canvas)
        axes.axis("off")
        if title is not None:
            fig.suptitle(title, fontsize=16)

        return fig, axes

    def apply(self, fn: tp.Callable) -> float | np.ndarray:
        """Apply a function to the pixel values of this instance."""
        return fn(self.masked())
