from __future__ import annotations

import os
from abc import ABC
from pathlib import Path

from dotenv import load_dotenv

from streetscapes import utils


class SourceBase(ABC):
    """TODO: Add docstrings"""

    def __init__(
        self,
        root_dir: str | Path | None = None,
    ):
        """
        A generic interface used by all derived
        interfaces to various data sources
        (HuggingFace, street view imagery, etc.)

        Args:
            root_dir:
                An optional custom root directory. Defaults to
                DATA_HOME/sources, where DATA_HOME is read from environment
                variables.
        """

        # Source name and environment variables
        # ==================================================
        self.name = self.__class__.__name__.lower()
        load_dotenv()

        # Root directory
        # ==================================================
        if root_dir is None:
            load_dotenv()
            data_home = os.getenv("DATA_HOME")
            root_dir = Path(data_home) / "sources" / self.name

        self.root_dir = utils.ensure_dir(root_dir)

        # An access token associated with this source
        # ==================================================
        env_prefix = utils.camel2snake(self.name).upper()
        self.token = os.getenv(f"{env_prefix}_TOKEN", None)

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
