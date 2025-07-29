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
                    raise TypeError("The operator is not callable.")

                subset = subset.filter(op(subset[lhs], rhs))

            if columns is not None:
                subset = subset.select(columns)

        return subset

    def fetch_image_urls(
            self, 
            table: ibis.Table,
            mp,
            kv 
    ) -> ibis.Table:
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
        
    def dowload_images(self, 
                       table: ibis.Table, 
                       mp, 
                       kv
    ) -> list[Path]:
        """Download images from Mapillary and KartaView."""
        paths = []
        df =  table.execute()
        for index, row in df.iterrows():
            if row["source"] == "Mapillary":
                path = mp.download_image(row["image_id"], row["image_url"])
                paths.append(path)
            elif row["source"] == "KartaView":
                path = kv.download_image(row["image_id"], row["image_url"])
                paths.append(path)
            else:
                logger.warning(f"Source not recognised for image {row['image_id']}.")
        return paths