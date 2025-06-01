# --------------------------------------
from __future__ import annotations

# --------------------------------------
from abc import ABC
from abc import abstractmethod

# --------------------------------------
import enum

# --------------------------------------
from environs import Env

# --------------------------------------
from pathlib import Path

# --------------------------------------
import typing as tp

# --------------------------------------
from streetscapes import utils


class SourceBase(ABC):
    """TODO: Add docstrings"""

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
    ):
        """
        A generic interface used by all derived
        interfaces to various data sources
        (HuggingFace, street view imagery, etc.)

        Args:
            env:
                An Env object containing loaded configuration options.

            root_dir:
                An optional custom root directory. Defaults to None.
        """

        # Source name and environment variable prefix
        # ==================================================
        self.name = self.__class__.__name__.lower()
        self.env_prefix = utils.camel2snake(self.name).upper()

        # Root directory
        # ==================================================
        root_dir = env.path(f"{self.env_prefix}_ROOT_DIR", root_dir)

        self.root_dir = utils.ensure_dir(root_dir)

        # An access token associated with this source
        # ==================================================
        self.token = env(f"{self.env_prefix}_TOKEN", None)

    def __repr__(self) -> str:
        """
        A printable representation of this class.
        """
        cls = self.__class__.__name__
        return f"{cls}(root_dir={utils.hide_home(self.root_dir)!r})"

    def show_contents(self) -> str | None:
        """
        Create and return a tree-like representation of a directory.
        """
        return utils.show_dir_tree(self.root_dir)
