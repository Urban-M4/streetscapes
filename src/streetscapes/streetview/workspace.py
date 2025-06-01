# --------------------------------------
import os

# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from environs import Env

# --------------------------------------
from streetscapes import utils
from streetscapes import logger


class SVWorkspace:
    """TODO: Add docstrings"""

    @staticmethod
    def restore(path: Path):
        """
        STUB
        A method to restore a workspace from a saved session.

        Args:
            path:
                The path to the workspace root directory.
        """
        return SVWorkspace()

    def __init__(
        self,
        path: Path | str,
        env: Path | str | None = None,
        create: bool = True,
    ):
        # Directories and paths
        # ==================================================
        # The root directory of the workspace
        path = Path(path)
        if not path.exists() and not create:
            raise FileNotFoundError("The specified path does not exist.")

        self.root_dir = utils.ensure_dir(path)

        # Add a .gitignore file to the root of the workspace
        # to avoid committing the workspace accidentally
        gitignore = self.root_dir / ".gitignore"
        gitignore.touch(mode=0o750)
        with open(gitignore, "w") as gfile:
            gfile.write("/**.*\n")

        # Configuration
        # ==================================================
        self.env = Env(expand_vars=True)
        if env is None and (local_env := self.root_dir / ".env").exists():
            env = local_env

        self.env.read_env(env)

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
