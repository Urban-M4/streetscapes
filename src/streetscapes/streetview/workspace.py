import os
from pathlib import Path

import ibis
from dotenv import load_dotenv

from streetscapes import utils
from streetscapes import logger


class SVWorkspace:
    """TODO: Add docstrings"""

    def __init__(
        self,
        name: str,
        create: bool = True,
    ):
        # Setup workspace directory
        load_dotenv()
        data_home = os.getenv("DATA_HOME", None)
        path = Path(data_home) / "workpaces" / name

        if not path.exists() and not create:
            raise FileNotFoundError("The specified path does not exist.")

        self.root_dir = utils.ensure_dir(path)

        # Metadata object.
        # Can be used to save and reload a workspace.
        # ==================================================
        self.metadata: ibis.BaseBackend = None

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(root_dir={utils.hide_home(self.root_dir)!r})"

    utils.exit_register
    def _cleanup(self):

        logger.info("Cleaning up...")
        if self.metadata is not None:
            self.metadata.disconnect()

    def load(self) -> ibis.BaseBackend:

        if self.metadata is None:

            metadata = self.root_dir / "metadata.ddb"
            self.metadata = ibis.duckdb.connect(f"{metadata}")

        return self.metadata

    def get_workspace_path(
        self,
        path: str | Path = None,
        suffix: str | None = None,
        create: bool = False,
    ):
        """
        Construct a workspace path (a file or a directory)
        with optional modifications.

        Args:
            path:
                The original path.
                Defaults to None.

            suffix:
                An optional (replacement) suffix. Defaults to None.

            create:
                Indicates that the path should be created if it doesn't exist.
                Defaults to False.

        Returns:
            The path to the file.
        """

        if path is None:
            path = self.root_dir

        path = self.root_dir / utils.make_path(
            path,
            self.root_dir,
            suffix=suffix,
        ).relative_to(self.root_dir)

        return (
            utils.ensure_dir(path) if create else path.expanduser().resolve().absolute()
        )

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

        filename = self.get_workspace_path(filename, suffix="csv")

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

        filename = self.get_workspace_path(filename, suffix="parquet")

        return ibis.read_parquet(filename)

    def show_contents(self) -> str | None:
        """
        Create and return a tree-like representation of a directory.
        """
        return utils.show_dir_tree(self.root_dir)
