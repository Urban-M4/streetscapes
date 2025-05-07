# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import typing as tp

# --------------------------------------
from streetscapes import utils


class SVInstance:
    """TODO: Add docstrings"""

    def __init__(
        self,
        image_path: Path,
        mask: np.ndarray,
        label: str,
        iid: int,
    ):
        self.image_path = image_path
        self.mask = mask
        self.label = label
        self.iid = iid

    def __repr__(self):
        return f"SVInstance(label={self.label})"

    def masked(
        self,
        channel: int = None,
        hsv: bool = False,
    ) -> np.ndarray:
        """
        Masked pixels corresponding to the instance.

        Args:
            channel:
                Only get the masked data for a specific channel.
                Defaults to None.

            hsv:
                Convert the original RBG image into HSV.
                Defaults to None.

        Returns:
            The pixels corresponding to the mask.
        """

        image = utils.open_image(self.image_path)
        if hsv:
            image = utils.as_hsv(image)

        image = image[self.mask[0], self.mask[1]]
        return image if channel is None else image[..., channel]

    def brightness(self) -> np.ndarray:
        """
        A convenience method to extract the brightness (= the value in HSV space)
        for the pixels corresponding to this instance.


        Returns:
            _description_:
                The per-pixel brightness of the instance.
        """

        return self.masked(hsv=True, channel=2)

    def visualise(
        self,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
        channel: int = None,
    ):
        """
        Visualise this instance only, with the rest of the image blanked out.

        Args:
            title: Add a title to the plot. Defaults to None.
            figsize: Figure size. Defaults to (16, 6).
            channel: Only plot the requested channel. Defaults to None.

        Returns:
            The instance plotted in isolation.
        """
        import matplotlib.pyplot as plt

        # Create a figure
        (fig, axes) = plt.subplots(1, 1, figsize=figsize)

        # Open the image and black out everything outside the mask
        image = utils.open_image(self.image_path)
        canvas = np.zeros_like(image)

        if isinstance(channel, int) and channel < len(image.shape):
            canvas[..., channel][self.mask[0], self.mask[1]] = image[..., channel][
                self.mask[0], self.mask[1]
            ]
        else:
            canvas[self.mask[0], self.mask[1]] = image[self.mask[0], self.mask[1]]

        axes.imshow(canvas)
        axes.axis("off")
        if title is not None:
            fig.suptitle(title, fontsize=16)

        return (fig, axes)

    def apply(self, fn: tp.Callable) -> float | np.ndarray:
        return fn(self.masked())
