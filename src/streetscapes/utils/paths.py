"""Filesystem and path utilities."""

import re
from pathlib import Path

import filetype as ft
import seedir as sd


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
