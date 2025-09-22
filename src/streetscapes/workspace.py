from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass
class Workspace:
    base_dir: Path

    @classmethod
    def from_env(cls):
        load_dotenv()
        base = os.getenv("STREETSCAPES_OUTPUT_DIR", "./streetscapes_output")
        base_dir = Path(base)
        base_dir.mkdir(parents=True, exist_ok=True)
        return cls(base_dir=base_dir)

    @property
    def manifests(self) -> Path:
        d = self.base_dir / "manifests"
        d.mkdir(parents=True, exist_ok=True)
        return d

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

    def read_manifest(self, name: str):
        """
        Read a manifest file from the manifests directory.
        Returns a pandas or geopandas DataFrame depending on file type.
        """
        path = self.manifests / name
        import geopandas as gpd

        return gpd.read_parquet(path)
