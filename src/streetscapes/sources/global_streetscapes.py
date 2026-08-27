"""Global Streetscapes related functionality."""

import operator
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import ibis
from huggingface_hub import hf_hub_download, scan_cache_dir, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

from streetscapes import logger, utils
from streetscapes.sources.base import SourceBase


class HFSourceBase(SourceBase, ABC):
    """TODO: Add docstrings."""

    @abstractmethod
    def load_dataset(
        self,
        criteria: dict | None = None,
        columns: list | tuple | set | None = None,
    ) -> ibis.Table:
        """Load and return a dataset.

        Args:
            criteria:
                Optional criteria used to create a subset.

            columns:
                The columns to keep or retrieve.

        Returns:
            An Ibis table.

        """
        pass

    def __init__(
        self,
        repo_id: str,
        repo_type: str,
        root_dir: Path | None = None,
    ):
        """A generic interface to a HuggingFace repository.

        Args:
            repo_id:
                HuggingFace repo ID.

            repo_type:
                HuggingFace repo type.

            root_dir:
                An optional custom root directory. Defaults to None.

        """
        # Repository details
        # ==================================================
        self.repo_id = repo_id
        self.repo_type = repo_type

        # Root directory
        # ==================================================
        # Ensure that the root directory is valid
        if root_dir is None:
            # Ensure that the HF cache directory exists
            if not (cache_path := Path(HF_HUB_CACHE)).exists():
                utils.ensure_dir(cache_path)

            # Scan the HF cache directory to extract the cached repos.
            cache = scan_cache_dir()
            for repo in cache.repos:
                if repo.repo_id == self.repo_id:
                    root_dir = repo.repo_path
                    break

            # If the repository hasn't been initialised yet,
            # we can construct the path manually.
            if root_dir is None:
                root_dir = Path(HF_HUB_CACHE) / repo_folder_name(
                    repo_id=self.repo_id,
                    repo_type=repo_type,
                )

        super().__init__(root_dir)

    def get_file(
        self,
        filename: str | Path,
    ) -> Path:
        """Retrieve a single (potentially cached) file from the Huggingface stored repo.

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
                local_dir=self.root_dir,
            )

        return Path(f)

    def get_files(
        self,
        filenames: list[str | Path],
    ) -> list[Path]:
        """Retrieve multiple (potentially cached) files from the HuggingFace repo.

        Args:
            filenames:
                The files to retrieve.

        Returns:
            A list of Path objects.

        """
        return [self.get_file(fname) for fname in filenames]


class GlobalStreetscapesSource(HFSourceBase):
    """TODO: Add docstrings."""

    def __init__(
        self,
        root_dir: Path | None = None,
    ):
        """An interface to the Global Streetscapes repository.

        Args:
            root_dir:
                An optional custom root directory. Defaults to None.

        """
        super().__init__(
            repo_id="NUS-UAL/global-streetscapes",
            repo_type="dataset",
            root_dir=root_dir,
        )

        # Paths for the Global Streetscapes cache directory and some
        # subdirectories for convenience.
        if self.root_dir is not None:
            self.csv_dir = self.root_dir / "data"
            self.parquet_dir = self.csv_dir / "parquet"

    def load_csv(
        self,
        filename: str | Path,
        root: Path | None = None,
    ) -> ibis.Table:
        """Load a CSV file from the Global Streetscapes repository.

        Args:
            filename:
                Name of the CSV file.

            root:
                Optional root directory. Defaults to None.

        Returns:
            An Ibis table.

        """
        fpath = utils.make_path(
            filename,
            root or self.csv_dir,
            suffix="csv",
        ).relative_to(self.root_dir)

        return ibis.read_csv(self.get_file(fpath))

    def load_parquet(
        self,
        filename: str | Path,
        root: Path | None = None,
    ):
        """Load a Parquet file from the Global Streetscapes repository.

        Args:
            filename:
                A Parquet file to load.

            root:
                Optional root directory. Defaults to None.

        Returns:
            An Ibis table.

        """
        fpath = utils.make_path(
            filename,
            root or self.parquet_dir,
            suffix="parquet",
        ).relative_to(self.root_dir)

        return ibis.read_parquet(self.get_file(fpath))

    def load_dataset(
        self,
        criteria: dict | None = None,
        columns: list | tuple | set | None = None,
    ) -> ibis.Table:
        """Load and return a dataset.

        Args:
            criteria:
                Optional criteria used to create a subset.

            columns:
                The columns to keep or retrieve.

        Returns:
            An Ibis table.

        """
        # Load the entire dataset
        gs_all = self.load_parquet("streetscapes")
        subset = gs_all

        if isinstance(criteria, dict):
            for lhs, criterion in criteria.items():
                if isinstance(criterion, (tuple, list, set)):
                    if len(criterion) > 2:
                        raise IndexError(f"Invalid criterion '{criterion}'")
                    op, rhs = (
                        (operator.eq, criterion[0])  # type: ignore[index]
                        if len(criterion) == 1
                        else criterion
                    )

                else:
                    op, rhs = operator.eq, criterion

                if not isinstance(op, tp.Callable):  # type: ignore
                    raise TypeError("The operator is not callable.")

                subset = subset.filter(op(subset[lhs], rhs))

            if columns is not None:
                subset = subset.select(columns)

        return subset

    def fetch_image_urls(self, table: ibis.Table, mp, kv) -> ibis.Table:
        """Fetch image URLs from Mapillary and KartaView."""
        df_urls = table.execute()
        for index, row in df_urls.iterrows():
            if row["source"] == "Mapillary":
                image_url = mp.get_image_url(row["image_id"])
                df_urls.at[index, "image_url"] = image_url
            elif row["source"] == "KartaView":
                image_url = kv.get_image_url(row["image_id"])
                df_urls.at[index, "image_url"] = image_url
            else:
                logger.warning(f"Source not recognised for image {row['image_id']}.")
        urls = ibis.memtable(df_urls)
        return urls

    def dowload_images(self, table: ibis.Table, mp, kv) -> list[Path]:
        """Download images from Mapillary and KartaView."""
        paths = []
        df = table.execute()
        for _, row in df.iterrows():
            if row["source"] == "Mapillary":
                path = mp.download_image(row["image_id"], row["image_url"])
                paths.append(path)
            elif row["source"] == "KartaView":
                path = kv.download_image(row["image_id"], row["image_url"])
                paths.append(path)
            else:
                logger.warning(f"Source not recognised for image {row['image_id']}.")
        return paths
