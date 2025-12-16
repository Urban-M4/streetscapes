from pathlib import Path

import ibis
from pandas import DataFrame
from shapely.geometry import box

from uuid import UUID

import numpy as np
import imageio as iio

from hashlib import sha256

from streetscapes import config
from streetscapes.utils import ensure_dir
from streetscapes.utils.bbox import Bbox


class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    def __init__(self, name: str | None = None):
        # TODO: also read name from config? But keep option to overwrite?
        self.name = name or "streetscapes"
        self.data_home = Path(config.get("data_home"))

        self.database_path = self.data_home / "projects" / f"{self.name}.duckdb"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.con = ibis.duckdb.connect(self.database_path)
        self.con.raw_sql("INSTALL spatial; LOAD spatial;")

        self.bootstrap()

    @property
    def core_tables(self) -> dict:
        """
        Core tables that need to be present in the project database.

        Returns:
            A dictionary of table names mapped to their schema.
        """

        return {
            "image_model": {
                "image_hash": ibis.dtype("!binary"),
                "model": ibis.dtype("!str"),
                "uuid": ibis.dtype("!uuid"),
            }
        }

    def get_image_dir(
        self,
        source: str | None = None,
        create: bool = False,
    ) -> Path:
        """
        Get the path to the directory where downloaded images are stored,
        optionally specifying a source.

        Args:
            model: The source name (e.g., 'mapillary').
            create: Optionally create the directory.

        Returns:
            A Path object.
        """
        path = self.data_home / "images"
        if source is not None:
            path /= source
        return ensure_dir(path) if create else path

    def get_output_dir(
        self,
        model: str,
        create: bool = False,
    ) -> Path:
        """
        Get the path to the output directory,
        optionally specifying a model.

        Args:
            model: The model name (e.g., 'maskformer').
            create: Optionally create the directory.

        Returns:
            A Path object.
        """
        path = self.data_home / "models"
        if model is not None:
            path /= model
        return ensure_dir(path) if create else path

    def ingest_mapillary(self, df: DataFrame, table: str = "mapillary"):
        """Ingest a DataFrame of Mapillary metadata."""
        self.con.con.register("metadata_tile", df)
        if table not in self.con.list_tables():
            self.con.raw_sql(
                f"""
                CREATE TABLE {table} AS
                SELECT
                    * EXCLUDE (geometry),
                    ST_GeomFromText(geometry) AS geometry,
                FROM metadata_tile;
                ALTER TABLE {table} ADD PRIMARY KEY (id);
            """
            )
        else:
            # TODO: consider configurable duplicate behaviour (REPLACE or IGNORE)
            self.con.raw_sql(
                f"""
                INSERT OR REPLACE INTO {table}
                SELECT
                    * EXCLUDE (geometry),
                    ST_GeomFromText(geometry) AS geometry,
                FROM metadata_tile
            """
            )

    def filter_bbox(self, table: str, bbox: Bbox):
        """Return an Ibis table expression filtered by a bounding box."""

        table = self.ensure_table(table)
        envelope_expr = ibis.literal(box(*bbox).wkt, type="geospatial:geometry")
        return table.filter(table.geometry.within(envelope_expr))

    # Export functions
    def _select_nonspatial(self, table: str) -> str:
        """Return SELECT clause converting geometry to WKT if present."""
        if "geometry" in self.con.table(table).columns:
            return "* EXCLUDE (geometry), ST_AsText(geometry) AS geometry"
        return "*"

    def _require_geometry(self, table: str, fmt: str):
        """Ensure table has a geometry column before geospatial export."""
        if "geometry" not in self.con.table(table).columns:
            raise ValueError(f"{fmt} export requires a 'geometry' column in '{table}'.")

    def export_parquet(self, table: str, output_path: str):
        self.con.raw_sql(f"COPY {table} TO '{output_path}' (FORMAT PARQUET);")

    def export_json(self, table: str, output_path: str):
        nonspatial = self._select_nonspatial(table)
        self.con.raw_sql(
            f"COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}' (FORMAT JSON);"
        )

    def export_csv(self, table: str, output_path: str):
        nonspatial = self._select_nonspatial(table)
        self.con.raw_sql(
            f"""COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}'
                (HEADER, DELIMITER ',');"""
        )

    def export_gpkg(self, table: str, output_path: str):
        self._require_geometry(table, "GeoPackage")
        self.con.raw_sql(
            f"COPY {table} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GPKG');"
        )

    def export_geojson(self, table: str, output_path: str):
        self._require_geometry(table, "GeoJSON")
        self.con.raw_sql(
            f"COPY {table} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GeoJSON');"
        )

    # TODO: could generalize to "get_records(table, columns, include='missing')"
    def get_mapillary_download_records(
        self, skip_existing: bool = True
    ) -> list[tuple[str, str]]:
        """Return list of (id, url) for Mapillary images to download."""
        base_query = "SELECT id, thumb_2048_url, ST_AsWKB(geometry) FROM mapillary"
        if skip_existing and "local_images" in self.con.list_tables():
            query = f"""
                {base_query}
                WHERE id NOT IN (
                    SELECT id FROM local_images WHERE source = 'mapillary'
                )
            """
        else:
            query = base_query

        return self.con.raw_sql(query).fetchall()

    def ingest_local_images(self, records: list[dict]):
        """Batch insert local images into `local_images`.

        Parameters
        ----------
        records : list of dict
            Each dict must have keys: 'id', 'source', 'path', 'geometry'.
            UUID will be generated inside DuckDB.

        """
        self.con.raw_sql(
            """
            CREATE TABLE IF NOT EXISTS local_images (
                image_hash TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                source TEXT NOT NULL,
                path TEXT NOT NULL,
                geometry GEOMETRY,
            )
        """
        )

        sql = """
        INSERT INTO local_images (image_hash, id, source, path, geometry)
        VALUES (GEN_RANDOM_UUID(), ?, ?, ?, ST_GeomFromWKB(?))
        ON CONFLICT DO NOTHING
        """

        params = [
            (r["id"], r["source"], str(r["path"]), r["geometry"]) for r in records
        ]

        # Execute as batch
        self.con.con.executemany(sql, params)

    def bootstrap(self):
        """
        Bootstrap the project with some core tables:

        - images: Images processed by *any* model.
            Serves as a reference table to check which images have been processed at all.
            Columns:
            - sha256 (can also serve as a unique ID)
            - geohash (see download_images.py)

        NOTE: In the schema definition, '!' in front of the type means 'non-nullable':
        https://ibis-project.org/reference/datatypes#parameters
        """

        # Tables that should exist in every project.
        # Just update the set with table names
        # and define the schema in the `schema` property.
        for name, schema in self.core_tables.items():
            self.ensure_table(name, schema)

    def ensure_table(
        self,
        name: str,
        schema: dict | ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ibis.Table:
        """
        Ensure that a table exists with the given schema.

        Args:
            name: Table name.
            schema: Schema to use for the table if it doesn't exist.
            overwrite: Overwrite the table if it exists.

        Returns:
            An Ibis table.
        """
        if name in self.con.tables and not overwrite:
            return self.con.table(name)
        if schema is None:
            if schema := self.core_tables.get(name) is None:
                raise ValueError(f"Please provide a valid schema for table '{name}'.")
        return self.con.create_table(name, schema=schema, overwrite=overwrite)

    def add_model_entries(
        self,
        image_hashes: bytes | list[bytes],
        models: str | list[str],
        uuids: UUID | list[UUID],
    ):
        """
        Add an entry to the intermediate lookup table.

        Args:
            image_hashes: SHA265 hashes of the processed images.
            models: Models used for processing the images.
            uuids: UUIDs of the entries in the model tables.
        """

        if isinstance(image_hashes, bytes):
            image_hashes = [image_hashes]

        if isinstance(models, str):
            models = [models]

        if isinstance(uuids, str):
            uuids = [uuids]

        data = {
            "image_hash": image_hashes,
            "model": models,
            "uuid": uuids,
        }

        # We assume that the images already exist in the local_images table:
        self.con.insert("image_model", data=data)

    def get_unprocessed_images(
        self,
        image_paths: list[Path],
        model: str,
    ) -> tuple[dict[bytes, UUID], list[tuple[bytes, UUID]]]:
        """
        Filter out processed images. Using the sha256 hash as the unique image ID.

        Args:
            image_paths: Image paths to process.
            model: The model to target.

        Returns:
            A list of paths to unprocessed image.
        """

        hashes = {
            sha256(np.asarray(iio.imread(path))).digest(): path for path in image_paths
        }

        t = self.con.table(model)
        processed = set(
            t.filter(t.image_hash.isin(list(hashes.keys())), t.model == model)
            .select("image_hash", "uuid")
            .to_pyarrow()
            .to_pydict()
        )

        processed = {h: u for h, u in zip(processed["image_hash"], processed["uuid"])}
        unprocessed = [(h, u) for h, u in hashes.items() if h not in processed]

        return processed, unprocessed
