import os
from pathlib import Path

import ibis
from dotenv import load_dotenv

import geopandas as gpd

from streetscapes import utils


class SVWorkspace:
    """TODO: Add docstrings"""

    def __init__(
        self,
        name: str,
        create: bool = True,
    ):
        load_dotenv()
        data_home = os.getenv("DATA_HOME", None)

        path = Path(data_home) / "workspaces" / name
        if not path.exists() and not create:
            raise FileNotFoundError("The specified path does not exist.")

        self.root_dir = utils.ensure_dir(path)

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(root_dir={utils.hide_home(self.root_dir)!r})"

    def get_workspace_path(
        self,
        filename: str | Path,
        suffix: str,
        create: bool = False,
    ):
        """
        Construct a workspace path (a file or a directory)
        with optional modifications.

        Args:
            path:
                The original path.

            suffix:
                File extension

            create:
                Indicates that the path should be created if it doesn't exist.
                Defaults to False.

        Returns:
            The path to the file.
        """
        return (self.root_dir / filename).with_suffix(suffix)

    def load_csv(
        self,
        filename: str | Path,
    ) -> ibis.Table:
        """
        Load a CSV file from the current workspace.

        Args:
            filename:
                The path to the file.

        Returns:
            An Ibis table.
        """

        filename = self.get_workspace_path(filename, suffix=".csv")

        return ibis.read_csv(filename)

    def load_parquet(
        self,
        filename: str | Path,
    ) -> ibis.Table:
        """
        Load a Parquet file from the current workspace.

        Args:
            filename:
                The path to the file.

        Returns:
            An Ibis table.
        """

        filename = self.get_workspace_path(filename, suffix=".parquet")

        return ibis.read_parquet(filename)

    def show_contents(self) -> str | None:
        """
        Create and return a tree-like representation of a directory.
        """
        return utils.show_dir_tree(self.root_dir)

    def save_metadata(
        self,
        records: gpd.GeoDataFrame,
        filename: str | Path = "metadata.parquet",
    ):
        """
        Save metadata to a Parquet file in the workspace.

        Args:
            records:
                The metadata records to save.
        """
        # TODO: Maybe use geoparquet? Or duckdb? Or postgis?

        filename = self.get_workspace_path(filename, suffix=".parquet", create=True)
        records.to_parquet(filename)

    def load_metadata(
        self,
        filename: str | Path = "metadata.parquet",
    ) -> gpd.GeoDataFrame:
        """
        Load metadata from a Parquet file in the workspace.

        Args:
            filename:
                The path to the metadata file.

        Returns:
            A GeoDataFrame containing the metadata.
        """
        filename = self.get_workspace_path(filename, suffix=".parquet")
        return gpd.read_parquet(filename)