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
from streetscapes.utils import CIEnum


class SourceType(CIEnum):
    """
    An enum listing supported sources.
    """

    GlobalStreetscapes = enum.auto()
    Mapillary = enum.auto()
    KartaView = enum.auto()
    Amsterdam = enum.auto()
    Google = enum.auto()


class SourceBase(ABC):
    """TODO: Add docstrings"""
    @staticmethod
    @abstractmethod
    def get_source_type() -> SourceType:
        """
        Get the enum corresponding to this source.
        """
        pass

    @staticmethod
    def _get_derived(
        parent: tp.Any | None = None,
        sources: dict | None = None,
    ) -> dict[SourceType, tp.Any]:

        if sources is None:
            sources = {}

        if parent is None:
            parent = SourceBase

        for source_cls in parent.__subclasses__():
            try:

                stype = source_cls.get_source_type()
                if stype is not None:
                    sources[stype] = source_cls
            except AttributeError as e:
                pass

            # Recurse
            SourceBase._get_derived(source_cls, sources)

        return sources

    @staticmethod
    def load_source(
        source_type: SourceType,
        env: Env,
        root_dir: str | Path | None = None,
    ) -> SourceBase:
        """
        Load a source.

        Args:
            source:
                An enum specifying the source to load.

            env:
                An Env object containing loaded configuration options.

            root_dir:
                An optional custom root directory. Defaults to None.

        Returns:
            The loaded source.
        """

        src = SourceBase._get_derived().get(source_type)
        if src is not None:
            return src(env, root_dir)

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
        self.name = self.get_source_type().name
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
