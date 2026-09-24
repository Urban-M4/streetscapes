"""Image loading and conversion utilities."""

from typing import TYPE_CHECKING, Any

from streetscapes.utils.logging import logger

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


# The orientation values the EXIF specification defines. Writers do emit
# nonsense (0 turns up in the wild), which is left alone rather than guessed at.
_ORIENTATIONS = frozenset(range(1, 9))

# The ones that turn an image through a quarter circle, swapping its axes.
TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})


def copy_upright(source: Path, target: Path) -> bool:
    """Copy an image, baking any EXIF orientation into the pixels themselves.

    A camera held sideways records landscape pixels plus an orientation tag.
    None of the readers used here (`imageio`, `skimage`, `PIL`) act on that tag,
    while browsers do, so a segmentation mask and the image shown next to it
    would disagree. Applying the rotation once, on import, settles it
    for every reader at once.

    Only the copy is rewritten; `source` is never touched.

    Args:
        source: The image to copy.
        target: Where to write the copy.

    Returns:
        True if the copy was turned upright, False if it was copied verbatim.
    """
    import shutil

    from PIL import Image, ImageOps

    with Image.open(source) as image:
        orientation = image.getexif().get(0x0112)  # ExifTags.Base.Orientation

        if orientation not in _ORIENTATIONS:
            if orientation is not None:
                logger.warning(
                    f"Ignoring the EXIF orientation of {source.name}: "
                    f"{orientation} is not one of the defined values 1-8."
                )
            shutil.copy2(source, target)
            return False

        if orientation == 1:
            # Already upright, so spare it a needless re-encode.
            shutil.copy2(source, target)
            return False

        upright = ImageOps.exif_transpose(image)

        # `exif_transpose` clears the orientation tag (in XMP as well), but
        # leaves the recorded pixel dimensions as they were before the turn.
        exif = upright.getexif()
        sub_ifd = exif.get_ifd(0x8769)  # ExifTags.IFD.Exif
        for tag, size in zip((0xA002, 0xA003), upright.size):  # PixelX/YDimension
            if tag in sub_ifd:
                sub_ifd[tag] = size

        # XMP is carried over too, or a panorama would lose its `GPano` properties.
        options: dict[str, Any] = {
            key: upright.info[key]
            for key in ("xmp", "icc_profile")
            if upright.info.get(key)
        }
        if exif:
            options["exif"] = exif.tobytes()

        if image.format == "JPEG":
            from PIL.JpegImagePlugin import get_sampling

            # Re-encode with the tables the image already used, which costs
            # neither noticeable quality nor extra bytes.
            options |= {
                "qtables": image.quantization,
                "subsampling": get_sampling(image),
            }

        upright.save(target, format=image.format, **options)

    return True


def upright_metadata(metadata: dict[str, Any]) -> None:
    """Rewrite EXIF metadata in place to describe an image turned upright.

    `copy_upright` bakes the rotation into the stored pixels, which leaves the
    orientation at 1 and, for a quarter turn, swaps the recorded dimensions.

    Args:
        metadata: EXIF metadata, as returned by `extract_exif_data`.
    """
    orientation = metadata.get("orientation")
    if orientation not in _ORIENTATIONS:
        return

    if orientation in TRANSPOSED_ORIENTATIONS:
        metadata["width"], metadata["height"] = metadata["height"], metadata["width"]

    metadata["orientation"] = 1
