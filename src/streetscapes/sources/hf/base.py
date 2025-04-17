# --------------------------------------
from abc import ABC
from abc import abstractmethod

# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from environs import Env

# --------------------------------------
from huggingface_hub import scan_cache_dir
from huggingface_hub import hf_hub_download
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

# --------------------------------------
from streetscapes import utils
from streetscapes.sources.base import SourceBase


class HFSourceBase(SourceBase, ABC):

    @abstractmethod
    def load_dataset(
        self,
        criteria: dict = None,
        columns: list | tuple | set = None,
    ) -> ibis.Table:
        """
        Load and return a dataset.

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
        env: Env,
        repo_id: str,
        repo_type: str,
        root_dir: Path | None = None,
    ):
        """
        A generic interface to a HuggingFace repository.

        Args:
            env:
                An Env object containing loaded configuration options.

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

        super().__init__(env, root_dir)

    def get_file(
        self,
        filename: str | Path,
    ) -> Path:
        """
        Retrieve a single (potentially cached) file
        from the Huggingface stored repo.

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
        """
        Retrieve multiple (potentially cached) files
        from the HuggingFace stored repo.

        Args:
            filenames:
                The files to retrieve.

        Returns:
            A list of Path objects.
        """
        return [self.get_file(fname) for fname in filenames]
