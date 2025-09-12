import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import json
from typing import List
from rich.progress import track


class ImageDownloader:
    """Generic image downloader for any source implementing `get_image_url` and `source_name`."""

    def __init__(
        self, source, output_dir: Path | str = "images", shard_size: int = 1000
    ):
        self.source = source
        self.shard_size = shard_size
        self.output_dir = Path(output_dir) / source.source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _shard_path(self, image_id: str) -> Path:
        # generate systematic folder names for 1000 images each
        hash_int = int(hashlib.md5(image_id.encode()).hexdigest(), 16)
        shard_folder = f"{hash_int % self.shard_size:04d}"
        return self.output_dir / shard_folder / f"{image_id}.jpg"

    def download(self, image_ids: List[str], overwrite=False):
        for image_id in track(
            image_ids, description=f"Downloading {self.source.source_name} images..."
        ):
            path = self._shard_path(image_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not overwrite:
                continue

            url = self.source.get_image_url(image_id)
            if not url:
                continue

            resp = self.source.session.get(url, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)

            # Record provenance
            self.manifest[image_id] = {
                "path": str(path),
                "downloaded_at": datetime.utcnow().isoformat(),
            }

        self._save_manifest()
