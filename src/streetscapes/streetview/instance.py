import numpy as np


class SVInstance:
    """TODO: Add docstrings"""
    def __init__(
        self,
        mask: np.ma.MaskedArray,
        label: str,
    ):
        self.mask = mask
        self.label = label

    def __repr__(self):
        return f"Instance(label={self.label})"

    def visualise(
        self,
        image: np.ndarray,
        title: str = None,
        figsize: tuple[int, int] = (16, 6),
    ):
        import matplotlib.pyplot as plt

        # Create a figure
        (fig, axes) = plt.subplots(1, 1, figsize=figsize)

        # Blank canvas
        canvas = np.zeros_like(image)

        # Overlay the instance
        canvas[self.mask] = image[self.mask]

        axes.imshow(canvas)
        axes.axis("off")
        if title is not None:
            fig.suptitle(title, fontsize=16)

        return (fig, axes)

