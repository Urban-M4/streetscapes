# --------------------------------------
from pathlib import Path

# --------------------------------------
import operator

# --------------------------------------
import ibis

# --------------------------------------
from environs import Env

# --------------------------------------
import typing as tp

# --------------------------------------
from streetscapes import utils
from streetscapes import logger
from streetscapes.sources.hf.base import HFSourceBase

class GlobalStreetscapesSource(HFSourceBase):
    """TODO: Add docstrings"""

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
    ):
        """
        An interface to the Global Streetscapes repository.

        Args:
            env:
                An Env object containing loaded configuration options.

            root_dir:
                An optional custom root directory. Defaults to None.
        """

        super().__init__(
            env,
            repo_id="NUS-UAL/global-streetscapes",
            repo_type="dataset",
            root_dir=root_dir,
        )

        # Paths for the Global Streetscapes cache directory and some
        # subdirectories for convenience.
        self.csv_dir = self.root_dir / "data" or None
        self.parquet_dir = self.csv_dir / "parquet" or None

    def load_csv(
        self,
        filename: str | Path,
        root: str | Path = None,
    ) -> ibis.Table:
        """
        Load a CSV file from the Global Streetscapes repository.

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
        root: str | Path = None,
    ):
        """
        Load a Parquet file from the Global Streetscapes repository.

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

        # Load the entire dataset
        gs_all = self.load_parquet("streetscapes")
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

                subset = subset.filter(op(subset[lhs], rhs))

            if columns is not None:
                subset = subset.select(columns)

        return subset


    def check_image_status(
        self,
        dataset: ibis.Table,
    ) -> tuple[set, set]:
        """
        Get the IDs of images that are missing from the local root directory.

        This method expects the colums corresponding to the source and the image ID
        to be named in a certain way (cf. self._source_col and self._id_col, respectively).
        This can be easily handled with Ibis by using .select() with a dictionary argument.
        For instance, assuming a table that contains columns named "source" and "orig_id"
        (as in the case of the Global Streetscapes dataset), we can obtain a new table
        with columns named "source" and "image_id" by passing a dictionary mapping the
        new column names to the existing ones:

        >>> t.select("source", "orig_id").columns
        ('source', 'orig_id')

        >>> t.select({'source': "source", "image_id": "orig_id"}).columns
        ('source', 'image_id')

        Here, 'source' is mapped unchanged to the original column called 'source'.

        Args:
            dataset:
                A dataset containing information about images that can be downloaded.

        Returns:
            A tuple containing:
                1. A set of existing images.
                2. A set of missing images.
        """

        sources = self.get_source_types_from_table(dataset)

        existing = {}
        missing = {}

        for src in sources:

            # Get the source object or add it if it's missing
            source = self.sources.get(src, self.add_source(src))
            if source is None:
                continue

            if isinstance(source, ImageSourceBase):

                filtered = [
                    str(s)
                    for s in dataset.filter(
                        dataset[self._source_col].ilike(f"%{src.name}")
                    )
                    .select(self._id_col)
                    .to_pandas()
                    .to_numpy()[:, 0]
                    .tolist()
                ]

                _existing, _missing = source.check_image_status(filtered)

                existing[src] = _existing
                missing[src] = _missing

        return existing, missing
