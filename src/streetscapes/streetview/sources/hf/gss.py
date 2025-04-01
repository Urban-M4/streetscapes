# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from huggingface_hub import CachedRepoInfo
from huggingface_hub import scan_cache_dir
from huggingface_hub import hf_hub_download

from huggingface_hub import try_to_load_from_cache

# --------------------------------------
from streetscapes.streetview.sources.hf.base import HFSourceBase
from streetscapes.streetview.workspace import SVWorkspace
from streetscapes.utils.logging import logger

class GlobalStreetscapesSource(HFSourceBase):

    def __init__(
        self,
        workspace: SVWorkspace,
    ):
        super().__init__(
            workspace,
            repo_id="NUS-UAL/global-streetscapes",
            repo_type="dataset",
            env_prefix="GLOBAL_STREETSCAPES",
        )

        # Paths for the Global Streetscapes cache and some
        # sub directoriesfor convenience.
        self.local_dir: Path | None = None
        self.csv_dir: Path | None = None
        self.parquet_dir: Path | None = None

    def _bootstrap(self):

        super()._bootstrap()

        # Load the info file.
        self.info = self.get_global_file_path("info.csv")

        # Now scan the cache directory to extract the paths
        cache_dir = scan_cache_dir()
        for repo in cache_dir.repos:
            if repo.repo_id == self.repo_id:
                self.local_dir = repo.repo_path

        # Bootstrap subdirectories
        # ==================================================
        self.csv_dir = self.local_dir / "data"
        self.parquet_dir = self.csv_dir / "parquet"

    def get_global_file_path(
        self,
        filename: str | Path,
    ):
        """
        Retrieve a single (potentially cached) file from the
        Global Streetscapes Huggingface dataset repo.

        Args:
            filename:
                The file to retrieve.

        Returns:
            A Path object.
        """

        # Ensure that we are not passing a path to the functions below.
        filename = str(filename)

        # Try to load the file from the cache.
        f = try_to_load_from_cache(
            filename=filename,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )

        if f is None:
            # Download the file
            f = hf_hub_download(
                filename=filename,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                local_dir=self.local_dir,
            )

        return Path(f)

    def get_file_paths(
        self,
        filenames: list[str | Path],
    ) -> list[Path]:
        """
        Retrieve multiple (potentially cached) files from the
        Global Streetscapes HuggingFace dataset repo.

        Args:
            filenames:
                The files to retrieve.

        Returns:
            A list of Path objects.
        """
        return [self.get_file_path(fname) for fname in filenames]

    def load_csv_file(
        self,
        filename: str | Path,
    ) -> ibis.Table:
        """
        Load a CSV file from the Global Streetscapes repository.

        Args:
            filename:
                A CSV file to load.

        Returns:
            An Ibis table.
        """

        fpath = self.construct_path(
            filename,
            self.csv_dir,
            self.local_dir,
            "csv",
        )

        return ibis.read_csv(self.get_global_file_path(fpath))

    def load_parquet_file(
        self,
        filename: str | Path,
    ):
        """
        Load a Parquet file from the Global Streetscapes repository.

        Args:
            filename:
                A Parquet file to load.

        Returns:
            An Ibis table.
        """

        fpath = self.construct_path(
            filename,
            self.parquet_dir,
            self.local_dir,
            "parquet",
        )

        return ibis.read_parquet(self.get_global_file_path(fpath))

    def load_subset(
        self,
        subset: str,
        criteria: dict = None,
        columns: list | tuple | set = None,
        recreate: bool = False,
        save: bool = True,
    ) -> ibis.Table:
        """
        Load and return a Parquet file for a specific city, if it exists.

        Args:
            subset:
                The subset to load.

            criteria:
                The criteria used to subset the Global Streetscapes dataset.

            columns:
                The columns to keep or retrieve.

            recreate:
                Recreate the subset if it exists.
                Defaults to False.

            save:
                Save a newly created subset.
                Defaults to True.

        Returns:
            An Ibis table.
        """

        fpath = self.construct_path(subset, suffix="parquet")

        if recreate or not fpath.exists():
            logger.info(f"Creating subset '{fpath.name}'...")

            # First, load the entire dataset
            gs_all = self.load_global_parquet_file("streetscapes")
            subset = gs_all

            if isinstance(criteria, dict):

                for lhs, criterion in criteria.items():

                    if isinstance(criterion, (tuple, list, set)):
                        if len(criterion) > 2:
                            raise IndexError(f"Invalid criterion '{criterion}'")
                        op, rhs = (
                            (operator.eq, criterion[0])
                            if len(criterion) == 1
                            else criterion
                        )

                    else:
                        op, rhs = operator.eq, criterion

                    if not isinstance(op, tp.Callable):
                        raise TypeError(f"The operator is not callable.")

                    subset = subset[op(subset[lhs], rhs)]

                if columns is not None:
                    subset = subset.select(columns)

                if save:
                    subset.to_parquet(fpath)
        else:
            logger.info(f"Loading '{fpath.name}'...")

            subset = self.load_parquet_file(fpath)
            if columns is not None:
                subset = subset.select(columns)

        logger.info(f"Done")

        return subset
