"""Image loading and conversion utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Delay slow imports for CLI responsiveness
    from pathlib import Path

    import numpy as np


def as_rgb(
    image: "np.ndarray",
    greyscale: bool = False,
) -> "np.ndarray":
    """Convert an image into an RGB version.

    Args:
        image:
            The image to convert.

        greyscale:
            Switch to convert the image to greyscale.
            Defaults to False.

    Returns:
        The RGB image.

    """
    import numpy as np
    import skimage as ski

    if len(image.shape) == 2:
        # The image is already greyscale.
        # Just convert it to RGB.
        image = ski.color.gray2rgb(image)

    else:
        if image.shape[-1] == 4:
            # Remove the alpha channel if it's present
            image = image[..., :-1]

        # Check if it needs to be converted to greyscale
        if greyscale:
            image = ski.color.gray2rgb(ski.color.rgb2gray(image))

    # Convert the image to ubyte
    image = ski.exposure.rescale_intensity(image, out_range=np.ubyte)

    return image


def as_hsv(image: "np.ndarray") -> "np.ndarray":
    """Convert an RGB image into HSV format.

    Args:
        image:
            The input RGB image.

    Returns:
        The HSV image.

    """
    import skimage as ski

    return ski.color.rgb2hsv(as_rgb(image))  # type: ignore


def open_image(
    path: Path,
    as_grey: bool = False,
) -> "np.ndarray":
    """Open an image as a NumPy array.

    Args:
        path:
            The path to the image file.
        as_grey:
            Open the image as a greyscale.

    Returns:
        A NumPy array containing the image.

    """
    import skimage as ski

    return ski.io.imread(path, as_grey)  # type: ignore[no-any-return]
