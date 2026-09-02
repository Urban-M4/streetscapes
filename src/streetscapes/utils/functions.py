"""Many utility functions."""

import os
import re
import sys
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import exifread
import filetype as ft
import numpy as np
import seedir as sd
import shapely
from dotenv import load_dotenv
from PIL import Image

from streetscapes.utils.metadata import ImageMeta

if sys.version_info >= (3, 14):
    from uuid import uuid7 as __uuid7
else:
    from uuid7gen import uuid7 as __uuid7


if TYPE_CHECKING:  # Delay slow imports for CLI responsiveness
    import geopandas as gpd
    import torch


def iso_timestamp(
    precision: str = "seconds",
    fmt: str | None = None,
    sep: str = "T",
    utc: bool = True,
) -> str:
    """Create a date-timestamp as a simplified ISO-formatted string.

    Useful for adding a unique but meaningful string to the
    name of a directory or a file that might be created
    repeatedly with the same name (for instance, when
    running the same experiment multiple times).
    The format is ISO 8601.

    NOTE: UTC time is used to avoid ambiguity.

    Args:
        precision: Precision for the timespec parameter.
        fmt: Explicit format.
        sep: A custom separator for the default ISO format.
        utc: Use UTC time (default)

    Returns:
        The formatted timestamp.
    """
    ts = datetime.now(UTC) if utc else datetime.now()

    if fmt is not None:
        return datetime.strftime(ts, fmt)
    tstr = ts.isoformat(sep=sep, timespec=precision)
    return tstr.split("+")[0]  # remove timezone info


def is_notebook() -> bool:
    """Determine if the caller is running in a Jupyter notebook.

    Courtesy of https://stackoverflow.com/a/39662359/4639195.

    Returns:
        bool:
            True if running in a notebook.

    """
    from IPython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        match shell:
            case "ZMQInteractiveShell":
                # Jupyter notebook or qtconsole
                return True
            case "TerminalInteractiveShell":
                # Terminal running IPython
                return False
            case _:
                # Other type (?)
                return False
    except NameError:
        # Probably standard Python interpreter
        return False


def ensure_dir(path: Path | str) -> Path:
    """Resolve and expand a directory path and create the directory if it doesn't exist.

    Args:
        path:
            A directory path.

    Returns:
        The (potentially newly created) expanded path.

    """
    path = Path(path).expanduser().resolve().absolute()
    path.mkdir(exist_ok=True, parents=True)
    return path


def hide_home(dir: Path) -> str:
    """A very simple function that replaces the home directory with a tilde.

    Useful for printing the home directory in notebooks without
    revealing private information.

    Args:
        dir:
            The directory to process.

    Returns:
        The directory with a tilde (~) instead of the user's home directory.

    """
    return str(dir).replace(str(Path.home()), "~")


def show_dir_tree(dir: Path) -> str | None:
    """Create and return a tree-like representation of a directory.

    TODO: Limit the depth, etc. Perhaps use **kwargs to pass options to `seedir.`

    Returns:
        The directory structure with the subdirectories and
        files that they contain.

    """
    return sd.seedir(  # type: ignore[no-any-return]
        dir,
        exclude_files=r"$(\.).*",
        exclude_folders=r"$(\.).*",
        regex=True,
    )


def filter_files(
    path: Path | str,
    pattern: str,
):
    """Filter files in a directory based on a pattern.

    Args:
        path:
            The path (a directory) to traverse.

        pattern:
            The regex pattern to apply.

    Raises:
        TypeError:
            Raised if a file is passed to the function.

    Returns:
        The filtered file paths.

    """
    if not (path := Path(path)).exists():
        return set()

    if path.is_file():
        raise TypeError("The provided path is a file (it should be a directory).")

    items = [str(n) for n in path.glob("*.*")]
    return {Path(p) for p in filter(re.compile(pattern, re.IGNORECASE).match, items)}


def make_path(
    path: str | Path,
    root: Path | None = None,
    suffix: str | None = None,
):
    """Construct a path (a file or a directory) with optional modifications.

    Args:
        path:
            The original path.

        root:
            An optional root path.
            Defaults to None.

        suffix:
            An optional (replacement) suffix. Defaults to None.

    Returns:
        The resolved path.

    """
    # Ensure that we have a Path object
    path = Path(path)

    # Optionally position the path relative to the root.
    if not path.is_absolute() and root is not None:
        path = root / path

    # Optionally replace or add a suffix.
    if suffix is not None:
        path = path.with_suffix(f".{suffix}")

    return path


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


def make_colourmap(
    labels: dict | list | tuple,
    cmap: str = "jet",
) -> dict:
    """Create a dictionary of colours (used for visualising instances).

    Args:
        labels:
            A dictionary of labels.

        cmap:
            Colourmap. Defaults to "jet".

    Returns:
        dict:
            Dictionary of class/colour associations.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(labels) == 0:
        return {}

    cm = plt.get_cmap(cmap, len(labels))
    cm = cm(np.linspace(0.0, 1.0, cm.N))[:, :3]  # type: ignore
    return dict(zip(sorted(labels), cm, strict=False))  # type: ignore


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


def camel2snake(string: str) -> str:
    """Convert a CamelCase string into a snake_case version.

    Args:
        string:
            The input CamelCase string.

    Returns:
        The output snake_case string.

    """
    # Replace each character with an underscore and its lowercase version:
    return "".join(
        [f"_{x.lower()}" if x.isupper() else x for x in string]
    ).removeprefix("_")


def get_env(key: str):
    """Read the value of `key` from the environment variables."""
    load_dotenv()
    value = os.getenv(key, None)

    if value is not None:
        return value

    raise KeyError(f"{key} not found in environment variables.")


def plot_metadata(gdf: "gpd.GeoDataFrame", ax=None):
    """Plot the metadata from a GeoDataFrame.

    Args:
        gdf:
            The GeoDataFrame containing the metadata.
        ax:
            The axes to plot on. Defaults to None.

    Returns:
        The axes with the plotted metadata.

    """
    import contextily as ctx

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(ax=ax, color="red", markersize=0.5, alpha=0.5)
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.nlmaps.standaard)
    return ax


def show_image(id: str, source: str):
    """Quickly plot an image.

    Args:
        id: The image ID.
        source: The source of the image (e.g., 'mapillary').

    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from PIL import Image

    image_dir = Path(get_env("DATA_HOME")) / "sources" / source / "images"
    image_path = image_dir / f"{id}.jpeg"

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{source}/{id}.jpeg")
    plt.show()


def extract_categories(
    prompt: str | list[str],
    as_list: bool = False,
) -> str | list[str]:
    """Extract labels (object categories) to look for from a free-form prompt.

    Args:
        prompt: The labels as a string or a list of strings.
            If a string is provided, the categories should be
            separated by commas or full stops.
        as_list: Return the prompt as a list of strings rather
            than joining all the strings together into a single prompt.

    Returns:
        A list of labels (object categories).
    """

    def flatten(xs: Iterable):
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    if not isinstance(prompt, str):
        prompt = ".".join(flatten(prompt))

    prompt = prompt.strip().lower()

    prompt = ".".join(
        [cat.strip() for cat in prompt.split(",") if len(cat.strip()) > 0]
    )
    prompt = ". ".join(
        [cat.strip() for cat in prompt.split(".") if len(cat.strip()) > 0]
    )

    if as_list:
        return [cat.strip() for cat in prompt.split(".") if len(cat.strip()) > 0]

    return f"{prompt.strip()}."


def get_device(device: "torch.device | str | None") -> "torch.device":
    """Get a Torch device.

    Args:
        device: A string / torch.device specification or None for a sane default.

    Returns:
        A torch.device object.
    """
    import torch

    if isinstance(device, torch.device):
        return device

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.mps.is_available() else "cpu")
        )
    return torch.device(device)


def get_image_hash(image: str | Path | bytes) -> bytes:
    """Get the SHA-256 hash of an image file.

    Args:
        image: The path to the file or raw bytes.

    Returns:
        SHA-256 digest.
    """
    if not ft.is_image(image):
        raise ValueError("The provided file is not an image.")

    if isinstance(image, bytes):
        import io

        image = io.BytesIO(image)  # type: ignore[assignment]

    return sha256(np.asarray(Image.open(image))).digest()


def hash2uuid(ihash: bytes) -> uuid.UUID:
    """Create a UUID (128 bits) from a SHA-256 hash of an image file.

    Args:
        ihash: The hash.

    Returns:
        A UUID.
    """
    return uuid.UUID(ihash.hex()[::2])


def get_image_uuid(image: str | Path | bytes) -> uuid.UUID:
    """Get the unique and reproducible UUID of an image file.

    Args:
        image: The path to the file or raw bytes.

    Returns:
        Image UUID.
    """
    if not ft.is_image(image):
        msg = "Input image type of is not supported!"
        raise ValueError(msg)

    return hash2uuid(get_image_hash(image))


def get_image_paths(path: str | Path) -> list[Path]:
    """Get only the image paths in a directory.

    Args:
        path: A directory of images.

    Returns:
        Image paths.
    """
    if not isinstance(path, Path | str):
        raise ValueError(f"Invalid path '{path}'")

    path = Path(path)
    if path.is_file():
        # Single file, return as list.
        return [path]

    entries = path.glob("**/*")
    image_paths = []
    for entry in entries:
        if not ft.is_image(entry):
            continue

        image_paths.append(entry)

    return image_paths


def get_image_metadata(image: bytes | str | Path) -> ImageMeta:
    """Get some reproducible image metadata.

    Args:
        image: Binary content or a path to an existing image.

    Returns:
        An object contiaining the image metadata.
    """
    _hash = get_image_hash(image)
    _uuid = hash2uuid(_hash)
    ext = ft.guess_extension(image).lower()

    if isinstance(image, (str, Path)):
        image = Path(image).read_bytes()

    return ImageMeta(image, _hash, _uuid, ext)


def get_geohash_shard_path(location: "shapely.Point"):
    """Get nested geo-hash path for given location given as a WKB point.

    Geo-hash precision from
    https://python-bloggers.com/2024/02/geohashing-from-scratch-in-python/
    Precision          Dimension
            1: 5,000km x 5,000km
            2:   1,250km x 625km
            3:     156km x 156km
            4:   31.9km x 19.5km
            5:   4.89km x 4.89km
            6:   1.22km x 0.61km
            7:       153m x 153m
            8:     38.2m x 19.1m
            9:     4.77m x 4.77m
           10:    1.19m x 0.596m
           11:     149mm x 149mm
           12:   37.2mm x 18.6mm
        Each level of precision subdivides the previous level into 32 subtiles.
    Shard path of precision 7, split in three parts abc/de/fg
    abc/ --> region level
    de/ --> neighbourhood scale (max 32x32 = 1024 per region)
    fg/ --> block level  (max 32x32 = 1024 per neighbourhood)
    """
    import pygeohash
    import shapely

    geom = shapely.from_wkb(location)  # type: ignore[call-overload]
    geohash = pygeohash.encode(geom.y, geom.x, precision=7)  # 153m x 153m
    return Path(geohash[:2]) / geohash[2:4] / geohash[4:6]


def uuid7(as_str: bool = False) -> uuid.UUID | str:
    """Return a UUID7 instance, optionally converted to string.

    Args:
        as_str: If True, convert the UUID to string before returning.

    Returns:
        The UUID.
    """
    u = __uuid7()
    return u if not as_str else str(u)


def to_deg(
    dms: list[int | float] | None,
    reference: Literal["N", "E", "S", "W"] | None = None,
) -> float:
    """Convert [deg, min, s] to degrees.

    Args:
        dms: A list containing degrees, minutes and seconds.
        reference: Optional coordinate reference (N, E, S, or W).

    Returns:
        Latitude or longitude coordinate in decimal degrees.
    """
    if dms is None:
        return 0.0

    sign = -1 if reference in ("W", "S") else 1
    deg = dms[0] + dms[1] / 60 + float(dms[2]) / 3600
    return sign * deg


def extract_exif_data(impath: Path) -> dict[str, Any]:
    """Extract EXIF metadata from an image file.

    Args:
        impath: Path to an image.
    """
    with open(impath, "rb") as file_handle:
        tags = exifread.process_file(file_handle)

    # Extract the tags that we are interested in
    lon = tags.get("GPS GPSLongitude")
    if lon is not None:
        lon = lon.values
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if lon_ref is not None:
        lon_ref = lon_ref.values
    lon = to_deg(lon, lon_ref)

    lat = tags.get("GPS GPSLatitude")
    if lat is not None:
        lat = lat.values
    lat_ref = tags.get("GPS GPSLatitudeRef")
    if lat_ref is not None:
        lat_ref = lat_ref.values
    lat = to_deg(lat, lat_ref)

    mapping = {
        "make": ("Image Make", str),
        "model": ("Image Model", str),
        "orientation": ("Image Orientation", int),
        "timestamp": (
            "Image DateTime",
            lambda x: datetime.strptime(x, "%Y:%m:%d %H:%M:%S"),
        ),
        "width": ("EXIF ExifImageWidth", int),
        "height": ("EXIF ExifImageLength", int),
        "altitude": ("GPS GPSAltitude", float),
        "compass_angle": ("GPS GPSTrack", float),
        "geometry": (shapely.Point([lon, lat]), None),
        "is_pano": (None, None),
        "iso": ("EXIF ISOSpeedRatings", int),
        "focal_length": ("EXIF FocalLength", float),
        "exposure": ("EXIF ExposureTime", float),
        "fstop": ("EXIF FNumber", float),
    }

    data = dict.fromkeys(mapping)

    for k, (val, caster) in mapping.items():
        if isinstance(val, str):
            val = tags.get(val)
            if val is not None:
                val = val.values

            if isinstance(val, list):
                val = val[-1]

        if val is None:
            continue

        data[k] = val if caster is None else caster(val)  # type: ignore[operator, assignment]

    return data
