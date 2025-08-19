import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import seedir as sd
from IPython import get_ipython
import skimage as ski
import geopandas as gpd

def is_notebook() -> bool:
    """Determine if the caller is running in a Jupyter notebook.

    Courtesy of https://stackoverflow.com/a/39662359/4639195.

    Returns:
        bool:
            True if running in a notebook.
    """
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
    """
    Resolve and expand a directory path and
    create the directory if it doesn't exist.

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
    """
    A very simple function that replaces the home directory
    with a tilde.

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
    """
    Create and return a tree-like representation of a directory.

    TODO: Limit the depth, etc. Perhaps use **kwargs to pass options to `seedir.`

    Returns:
        The directory structure with the subdirectories and
        files that they contain.
    """
    return sd.seedir(
        dir,
        exclude_files=r"$(\.).*",
        exclude_folders=r"$(\.).*",
        regex=True,
    )


def filter_files(
    path: Path | str,
    pattern: str,
):
    """
    Filter files in a directory based on a pattern.

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
    return set(
        [Path(p) for p in filter(re.compile(pattern, re.IGNORECASE).match, items)]
    )


def make_path(
    path: str | Path,
    root: Path | None = None,
    suffix: str | None = None,
):
    """
    Construct a path (a file or a directory)
    with optional modifications.

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
    image: np.ndarray,
    greyscale: bool = False,
) -> np.ndarray:
    """
    Convert an image into an RGB version.

    Args:
        image:
            The image to convert.

        greyscale:
            Switch to convert the image to greyscale.
            Defaults to False.

    Returns:
        The RGB image.
    """

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


def as_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image into HSV format

    Args:
        image:
            The input RGB image.

    Returns:
        The HSV image.
    """

    return ski.color.rgb2hsv(as_rgb(image))


def make_colourmap(
    labels: dict | list | tuple,
    cmap: str = "jet",
) -> dict:
    """
    Create a dictionary of colours (used for visualising instances).

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

    if len(labels) == 0:
        return {}

    cmap = plt.get_cmap(cmap, len(labels))
    cmap = cmap(np.linspace(0.0, 1.0, cmap.N))[:, :3]
    return {label: colour for label, colour in zip(sorted(labels), cmap)}


def open_image(
    path: Path,
    as_grey: bool = False,
) -> np.ndarray:
    """
    Open an image as a NumPy array.

    Args:
        path:
            The path to the image file.
        as_grey:
            Open the image as a greyscale.

    Returns:
        A NumPy array containing the image.
    """

    return ski.io.imread(path, as_grey)


def camel2snake(string: str) -> str:
    """
    Convert a CamelCase string into a snake_case version.

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
    """Read the value of `key` from the environment variables.

    Environment variables may be set in .env or defined in current shell.
    """
    load_dotenv()
    value = os.getenv(key, None)

    if value is not None:
        return value

    raise KeyError(f"{key} not found in environment variables.")


def plot_metadata(gdf: gpd.GeoDataFrame, ax=None):
    """
    Plot the metadata from a GeoDataFrame.

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

    from PIL import Image
    import matplotlib.pyplot as plt

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
