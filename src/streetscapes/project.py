from pathlib import Path

import ibis
from pandas import DataFrame
from shapely.geometry import box

from uuid import UUID
import duckdb
import platformdirs as pdirs
import shutil

from streetscapes import utils
from streetscapes import config
from streetscapes import logger
from streetscapes.utils import ensure_dir
from streetscapes.utils.bbox import Bbox


class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    # Core tables that need to be present in the project database.
    core_tables = {
        "images": {
            "schema": {
                "hash": "BINARY PRIMARY KEY",
                "source": "STRING NOT NULL",
                "path": "STRING NOT NULL",
                "tags": "STRING[]",
            },
            "init": [],
        },
        "collections": {
            "schema": {
                "name": "STRING NOT NULL",
                "hash": "BINARY NOT NULL",
            },
            "init": ["ALTER TABLE collections ADD PRIMARY KEY (name, hash);"],
        },
        "segmentations": {
            "schema": {
                "collection": "STRING NOT NULL",
                "model": "STRING NOT NULL",
                "run": "STRING NOT NULL",
                "archive": "STRING NOT NULL",
                "params": "BINARY",
            },
            "init": [
                "ALTER TABLE segmentations ADD PRIMARY KEY (collection, model, run);"
            ],
        },
        "mapillary": {
            "schema": {
                "altitude": "FLOAT8",
                "atomic_scale": "FLOAT8",
                "camera_type": "STRING",
                "camera_parameters": "FLOAT8[]",
                "captured_at": "UBIGINT",
                "compass_angle": "FLOAT8",
                "computed_altitude": "FLOAT8",
                "computed_compass_angle": "FLOAT8",
                "computed_geometry": "GEOMETRY",
                "computed_rotation": "FLOAT8[]",
                "creator": "JSON",
                "exif_orientation": "UBIGINT",
                "geometry": "GEOMETRY",
                "height": "UBIGINT",
                "id": "UBIGINT PRIMARY KEY",
                "is_pano": "BOOL",
                "make": "STRING",
                "model": "STRING",
                "sequence": "STRING",
                "thumb_1024_url": "STRING",
                "thumb_2048_url": "STRING",
                "thumb_256_url": "STRING",
                "thumb_original_url": "STRING",
                "width": "UBIGINT",
                "camera_parameters": "FLOAT8[]",
            },
            "init": [],
        },
        # TODO: KartaView and Amsterdam tables
    }

    def __init__(
        self,
        name: str | None = None,
        data_dir: str | Path | None = None,
        root_dir: str | Path | None = None,
    ):

        # TODO: also read name from config? But keep option to overwrite?
        self.name = name or "streetscapes"

        # Ensure that the root directory exists
        self.root_dir = ensure_dir(
            config.get(
                "root_dir",
                root_dir or pdirs.user_data_path("streetscapes"),
            )
        )
        self.project_home = Path(
            config.get("project_home", ensure_dir(self.root_dir / "projects"))
        )

        # Directory for cached data (images)
        self.data_home = Path(
            config.get(
                "data_home",
                ensure_dir(
                    data_dir or pdirs.user_cache_path("streetscapes"),
                ),
            )
        )

        config.set("active_project", self.name)

        # Internal attributes.
        # ==================================================
        self._db_name = "metadata"
        self._con = ibis.duckdb.connect(
            self.db_path,
            extensions=["spatial", "json"],
        )

        self._bootstrap()
        self._con.raw_sql(f"USE {self._db_name};")

    @property
    def db_path(self) -> Path:
        return self.project_home / f"{self.name}.duckdb"

    @property
    def archive_path(self) -> Path:
        return self.project_home / "archives"

    @property
    def image_path(self) -> Path:
        return self.data_home / "images"

    def _bootstrap(self):
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

        if self._db_name in self._con.list_databases():
            return

        self._con.create_database(self._db_name)

        # Tables that should exist in every project.
        # Just update the set with table names
        # and define the schema in the `schema` property.
        for name, items in self.core_tables.items():
            self.ensure_table(name)

            if (init := items.get("init")) is not None:
                for sql in init:
                    self._con.raw_sql(sql)

    def ensure_table(
        self,
        name: str,
        schema: dict | ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ibis.Table:
        """
        Ensure that a table exists with the given schema.

        TODO: Convert raw SQL into Ibis expressions.

        Args:
            name: Table name.
            schema: Schema to use for the table if it doesn't exist.
            overwrite: Overwrite the table if it exists.

        Returns:
            An Ibis table.
        """
        if name in self._con.tables and not overwrite:
            return self._con.table(name)
        if schema is None:
            table = self.core_tables.get(name)
            if table is None:
                raise ValueError(f"Please provide a valid schema for table '{name}'.")
            if (schema := table.get("schema")) is None:
                raise ValueError(f"Please provide a valid schema for table '{name}'.")

        if overwrite:
            sql = f"CREATE OR REPLACE TABLE {name}"
        else:
            sql = f"CREATE TABLE IF NOT EXISTS {name}"

        fields = ", ".join(
            [f"{fname} {definition}" for fname, definition in schema.items()]
        )
        sql = f"{sql} ({fields});"

        if overwrite and name in self._con.tables:
            self._con.drop_table(name, database=self._db_name)

        return self._con.raw_sql(sql)

    def get_image_path(
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
        path = self.image_path
        if source is not None:
            path /= source
        return ensure_dir(path) if create else path

    def get_archive_path(
        self,
        model: str,
        create: bool = False,
    ) -> Path:
        """
        Get the path to the archive directory,
        optionally specifying a model.

        Args:
            model: The model name (e.g., 'maskformer').
            create: Optionally create the directory.

        Returns:
            A Path object.
        """
        path = self.archive_path
        if model is not None:
            path /= model
        return ensure_dir(path) if create else path

    def get_archive_uuid(
        self,
        collection: str,
        model: str,
        run: str,
        create: bool = False,
    ) -> UUID | None:
        """
        Get the archive UUID from a (collection, model, run) key.

        If the key doesn't exist, a random UUID is returned if
        `create` is True, otherwise None.

        Args:
            collection: Collection name.
            model: Model name.
            run: The model urn.
            create: Create a random UUID if it's missing.

        Returns:
            UUID of the archive.
        """
        tbl = self._con.table("segmentations")
        archive_id = (
            tbl.filter(
                [
                    tbl.collection == collection,
                    tbl.model == model,
                    tbl.run == run,
                ]
            )
            .select("archive")
            .to_pyarrow()
            .to_pydict()["archive"]
        )

        if len(archive_id) == 0:
            if create:
                return utils.uuid7()
            return
        return archive_id[0]

    def ingest_mapillary(self, df: DataFrame, table: str = "mapillary"):
        """Ingest a DataFrame of Mapillary metadata."""
        self._con.con.register("metadata_tile", df)
        # if table not in self._con.list_tables():
        #     self._con.raw_sql(
        #         f"""
        #         CREATE TABLE {table} AS
        #         SELECT
        #             * EXCLUDE (geometry),
        #             ST_GeomFromText(geometry) AS geometry,
        #         FROM metadata_tile;
        #         ALTER TABLE {table} ADD PRIMARY KEY (id);
        #     """
        #     )
        # else:
        # TODO: consider configurable duplicate behaviour (REPLACE or IGNORE)
        self._con.raw_sql(f"INSERT OR REPLACE INTO {table} FROM metadata_tile")

        # SELECT
        #     * EXCLUDE (geometry),
        #     ST_GeomFromText(geometry) AS geometry,

    def filter_bbox(self, table: str, bbox: Bbox):
        """Return an Ibis table expression filtered by a bounding box."""

        table = self.ensure_table(table)
        envelope_expr = ibis.literal(box(*bbox).wkt, type="geospatial:geometry")
        return table.filter(table.geometry.within(envelope_expr))

    # Export functions
    def _select_nonspatial(self, table: str) -> str:
        """Return SELECT clause converting geometry to WKT if present."""
        if "geometry" in self._con.table(table).columns:
            return "* EXCLUDE (geometry), ST_AsText(geometry) AS geometry"
        return "*"

    def _require_geometry(self, table: str, fmt: str):
        """Ensure table has a geometry column before geospatial export."""
        if "geometry" not in self._con.table(table).columns:
            raise ValueError(f"{fmt} export requires a 'geometry' column in '{table}'.")

    def export_parquet(self, table: str, output_path: str):
        self._con.raw_sql(f"COPY {table} TO '{output_path}' (FORMAT PARQUET);")

    def export_json(self, table: str, output_path: str):
        nonspatial = self._select_nonspatial(table)
        self._con.raw_sql(
            f"COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}' (FORMAT JSON);"
        )

    def export_csv(self, table: str, output_path: str):
        nonspatial = self._select_nonspatial(table)
        self._con.raw_sql(
            f"""COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}'
                (HEADER, DELIMITER ',');"""
        )

    def export_gpkg(self, table: str, output_path: str):
        self._require_geometry(table, "GeoPackage")
        self._con.raw_sql(
            f"COPY {table} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GPKG');"
        )

    def export_geojson(self, table: str, output_path: str):
        self._require_geometry(table, "GeoJSON")
        self._con.raw_sql(
            f"COPY {table} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GeoJSON');"
        )

    # TODO: could generalize to "get_records(table, columns, include='missing')"
    def get_mapillary_download_records(
        self, skip_existing: bool = True
    ) -> list[tuple[str, str]]:
        """Return list of (id, url) for Mapillary images to download."""

        self.ensure_table("mapillary")
        base_query = "SELECT id, thumb_2048_url, ST_AsWKB(geometry) FROM mapillary"
        if skip_existing and "images" in self._con.list_tables():
            query = f"""
                {base_query}
                WHERE id NOT IN (
                    SELECT id FROM images WHERE source = 'mapillary'
                )
            """
        else:
            query = base_query

        return self._con.raw_sql(query).fetchall()

    def ingest_images(self, records: list[dict]):
        """Batch insert local images into `images`.

        Parameters
        ----------
        records : list of dict
            Each dict must have keys: 'id', 'source', 'path', 'geometry'.
        """

        # self._con.raw_sql(
        #     """
        #     CREATE TABLE IF NOT EXISTS images (
        #         hash TEXT PRIMARY KEY,
        #         id TEXT NOT NULL,
        #         source TEXT NOT NULL,
        #         path TEXT NOT NULL,
        #         geometry GEOMETRY
        #     )
        # """
        # )

        sql = """
        INSERT INTO images (id, source, path, geometry)
        VALUES (GEN_RANDOM_UUID(), ?, ?, ?, ST_GeomFromWKB(?))
        ON CONFLICT DO NOTHING
        """

        params = [
            (r["id"], r["source"], str(r["path"]), r["geometry"]) for r in records
        ]

        # Execute as batch
        self._con.con.executemany(sql, params)

    def add_segmentations(
        self,
        collections: str | list[str],
        models: str | list[str],
        runs: str | list[str],
        archives: UUID | list[UUID],
    ):
        """
        Add an entry to the intermediate lookup table.

        Args:
            collections: Collection names.
            models: Models used for processing the images.
            runs: Model runs.
            archives: UUIDs of the entries in the model tables.
        """

        if isinstance(collections, str):
            collections = [collections]

        if isinstance(models, str):
            models = [models]

        if isinstance(runs, str):
            runs = [runs]

        if isinstance(archives, UUID):
            archives = [archives]

        data = {
            "collection": [collections],
            "model": [models],
            "run": [runs],
            "archive": [ibis.uuid(a).to_pyarrow() for a in archives],
        }

        # We assume that the images already exist in the `images`` table:
        self._con.insert("segmentations", data)

    def get_segmentation_status(
        self,
        collection: str,
        model: str,
        run: str,
    ) -> tuple[dict[bytes, UUID], list[tuple[bytes, UUID]]]:
        """
        Filter out processed images. Using the sha256 hash as the unique image ID.

        Args:
            image_paths: Image paths to process.
            model: The model to target.
            run: The run associated with this model.

        Returns:
            A list of paths to unprocessed image.
        """

        im = self._con.table("images")
        col = self._con.table("collections")
        seg = self._con.table("segmentations")
        seg_filtered = seg.filter(
            [
                seg.collection == collection,
                seg.model == model,
                seg.run == run,
            ]
        )

        archive = seg_filtered.select("archive").to_pyarrow().to_pydict()["archive"]
        if len(archive) > 0:
            archive = archive[0]
        else:
            archive = utils.uuid7()
            self.add_segmentations(collection, model, run, archive)

        print(f"==[ archive: {archive}")
        archive_path = ensure_dir(self.get_archive_path(model) / str(archive))
        print(f"==[ archive path: {archive_path} | exists: {archive_path.exists()}")

        t_im = seg_filtered.inner_join(im, seg_filtered.hash == im.hash)
        t = t_im.outer_join(seg, t_im.name == seg.collection)

        print(t)
        print(t.columns)

        keys = ("hash", "path", "archive")
        entries = t.filter([]).select(*keys).to_pyarrow().to_pydict()

        # First half of each hash
        u2h = {
            str(utils.hash2uuid(h)): h
            for h in t.select("hash").to_pyarrow().to_pydict()["hash"]
        }

        processed = {}

        processed.update({u2h[p.stem]: p.stem for p in archive_path.iterdir()})

        unprocessed = {
            h: p for h, p in zip(entries["hash"], entries["path"]) if h not in processed
        }
        return processed, unprocessed

    def register_image(
        self,
        path: Path,
        source: str,
        remove: bool = False,
    ):
        """
        Register a downloaded (local) image into the database.

        Args:
            path: Path to the image file.
            source: The source of the image (e.g., 'mapillary').
            remove: Remove if the image is a duplicate.
        """

        ihash = utils.get_image_hash(path)

        if ihash is None:
            return

        if "images" not in self._con.tables:
            logger.warning(
                f"Project database seems to be corrupted: missing table 'images'."
            )
            return

        data = {
            "hash": [ihash],
            "source": [source],
            "path": [path.name],
        }

        try:
            self._con.insert("images", data)

        except duckdb.ConstraintException:
            logger.warning(
                f"Error registering {path.name}: duplicate key, moving on..."
            )
            if remove:
                path.unlink()

    def register_collection(
        self,
        collection: str,
        paths: list[Path],
        overwrite: bool = False,
    ):
        """
        Register a downloaded (local) image into the database.

        Args:
            collection: The collection to add the images to.
            paths: Paths to the image files.
            overwrite: Overwrite existing entries.
        """

        ihashes = [utils.get_image_hash(path) for path in paths]
        ihashes = [h for h in ihashes if h is not None]

        if len(ihashes) == 0:
            return

        if "collections" not in self._con.tables:
            logger.warning(
                f"Project database seems to be corrupted: missing table 'collections'."
            )
            return

        data = {
            "name": [collection for _ in range(len(ihashes))],
            "hash": ihashes,
        }

        try:
            self._con.insert("collections", data, overwrite=overwrite)

        except duckdb.ConstraintException:
            logger.warning(
                f"Error registering collection '{collection}': duplicate key..."
            )
