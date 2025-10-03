from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
import duckdb

from streetscapes.sources.mapillary import MapillaryDBWriter


@dataclass
class Workspace:
    """
    A workspace manages the output directories and provides access
    to the central DuckDB database for manifests.
    """
    base_dir: Path

    DB_NAME = "streetscapes.duckdb"

    @classmethod
    def from_env(cls):
        load_dotenv()
        base = os.getenv("STREETSCAPES_OUTPUT_DIR", "./streetscapes_output")
        base_dir = Path(base)
        base_dir.mkdir(parents=True, exist_ok=True)
        return cls(base_dir=base_dir)

    @property
    def db_path(self) -> Path:
        """Path to the central DuckDB workspace database."""
        return self.base_dir / self.DB_NAME

    @property
    def images(self) -> Path:
        d = self.base_dir / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def segmentation(self) -> Path:
        d = self.base_dir / "segmentation"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def footprints(self) -> Path:
        d = self.base_dir / "footprints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache(self) -> Path:
        d = self.base_dir / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs(self) -> Path:
        d = self.base_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----------------------
    # Database / writers
    # ----------------------
    def mapillary_writer(self, table_name: str = "mapillary") -> MapillaryDBWriter:
        """
        Provide a MapillaryDBWriter connected to this workspace database.
        """
        return MapillaryDBWriter(db_path=self.db_path, table_name=table_name)

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        """Return a DuckDB connection to the workspace DB."""
        return duckdb.connect(self.db_path)

    # ----------------------
    # Legacy / helper
    # ----------------------
    def show_tree(self):
        for subdir in [
            self.manifests,
            self.images,
            self.segmentation,
            self.footprints,
            self.cache,
            self.logs,
        ]:
            print(subdir)

    # deprecated for DB-centric manifests
    # def read_manifest(self, name: str):
    #     path = self.manifests / name
    #     import geopandas as gpd
    #     return gpd.read_parquet(path)
