import hashlib
import shutil
from pathlib import Path
import datetime
from rich.progress import track


class ImageDownloader:
    def __init__(
        self, source, manifest_dir: Path, images_dir: Path, shard_size: int = 1000
    ):
        import ibis
        self.source = source
        self.shard_size = shard_size
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = manifest_dir

        self.path = self.manifest_dir / "download_manifest.duckdb"
        self.con = ibis.duckdb.connect(str(self.path))
        if "downloads" not in self.con.list_tables():
            self.con.raw_sql("""
                CREATE TABLE downloads (
                    image_id VARCHAR,
                    path VARCHAR,
                    downloaded_at TIMESTAMP,
                    url VARCHAR
                )
            """)

    def _shard_path(self, image_id: str, index: int = None) -> Path:
        # Use sequential sharding: group images into folders of shard_size
        if index is not None:
            shard_folder = f"{index // self.shard_size:04d}"
        else:
            # fallback to hash-based if index not provided
            hash_int = int(hashlib.md5(image_id.encode()).hexdigest(), 16)
            shard_folder = f"{hash_int % self.shard_size:04d}"
        return self.images_dir / shard_folder / f"{image_id}.jpg"

    def _is_downloaded(self, image_id: str) -> bool:
        result = self.con.raw_sql(
            f"SELECT 1 FROM downloads WHERE image_id = '{image_id}' LIMIT 1"
        ).fetchall()
        return bool(result)

    def export_manifest(self):
        """
        Export the downloads table to a Parquet manifest for downstream use using ibis' to_parquet().
        Delete the DuckDB file after export to avoid remnants.
        """
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(self.manifest_dir / "downloaded_images.parquet")
        self.con.table("downloads").to_parquet(output_file)
        # Close and delete DuckDB file
        self.con.disconnect()
        if self.path.exists():
            self.path.unlink()
        return output_file

    def download_by_id(self, image_ids, overwrite=False):
        for idx, image_id in enumerate(
            track(image_ids, description="Downloading images by ID...")
        ):
            if not overwrite and self._is_downloaded(image_id):
                continue
            path = self._shard_path(image_id, index=idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            url = self.source.get_image_url(image_id)
            if not url:
                continue
            resp = self.source.session.get(url, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            safe_id = str(image_id).replace("'", "''")
            safe_path = str(path).replace("'", "''")
            safe_url = str(url).replace("'", "''")
            safe_timestamp = timestamp.replace("'", "''")
            sql = f"INSERT INTO downloads VALUES ('{safe_id}', '{safe_path}', '{safe_timestamp}', '{safe_url}')"
            self.con.raw_sql(sql)
        self.export_manifest()

    def download_from_manifest(
        self, manifest_df, id_column, url_column, overwrite=False
    ):
        for idx, (_, row) in enumerate(
            track(
                manifest_df.iterrows(),
                description=f"Downloading images from {url_column}...",
            )
        ):
            image_id = str(row[id_column])
            if not overwrite and self._is_downloaded(image_id):
                continue
            url = row.get(url_column)
            if not url:
                continue
            path = self._shard_path(image_id, index=idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            resp = self.source.session.get(url, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            # Interpolate values directly into SQL string
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            # Escape single quotes in strings
            safe_id = str(image_id).replace("'", "''")
            safe_path = str(path).replace("'", "''")
            safe_url = str(url).replace("'", "''")
            safe_timestamp = timestamp.replace("'", "''")
            sql = f"INSERT INTO downloads VALUES ('{safe_id}', '{safe_path}', '{safe_timestamp}', '{safe_url}')"
            self.con.raw_sql(sql)
        self.export_manifest()