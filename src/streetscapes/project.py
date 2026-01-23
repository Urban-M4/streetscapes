from pathlib import Path

import ibis
from pandas import DataFrame
from shapely.geometry import box

from uuid import UUID
import duckdb
import platformdirs as pdirs

from streetscapes import utils
from streetscapes import config
from streetscapes import logger
from streetscapes.utils.bbox import Bbox
from streetscapes.utils.metadata import ImageMeta


class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    # Core tables that need to be present in the project database.
    core_tables = {
        "images": {
            "schema": {
                "uuid": "UUID PRIMARY KEY",
                "source": "STRING",
                "shard": "STRING",  # An optional path relative to the main image directory
                "notes": "STRING",
                "tags": "STRING[]",
                "rating": "INTEGER",  # 0-5
            },
            "init": [],
        },
        "collections": {
            "schema": {
                "name": "STRING NOT NULL",
                "image": "UUID NOT NULL",
            },
            "init": ["ALTER TABLE collections ADD PRIMARY KEY (name, image);"],
        },
        "labels": {
            "schema": {
                "image": "UUID NOT NULL",
                "model": "STRING NOT NULL",
                "run": "STRING NOT NULL",
                "labels": "STRING[] NOT NULL",
            },
            "init": ["ALTER TABLE labels ADD PRIMARY KEY (image, model, run);"],
        },
        "segmentations": {
            "schema": {
                "collection": "STRING NOT NULL",
                "model": "STRING NOT NULL",
                "run": "STRING NOT NULL",
                "archive": "UUID NOT NULL",  # UUID7
                "params": "BINARY",
            },
            "init": [
                "ALTER TABLE segmentations ADD PRIMARY KEY (collection, model, run);"
            ],
        },
        "mapillary": {
            "schema": {
                "uuid": "UUID",
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
        self.root_dir = utils.ensure_dir(
            config.get(
                "root_dir",
                root_dir or pdirs.user_data_path("streetscapes"),
            )
        )
        self.project_home = Path(
            config.get("project_home", utils.ensure_dir(self.root_dir / "projects"))
        )

        # Directory for cached data (images)
        self.data_home = Path(
            config.get(
                "data_home",
                utils.ensure_dir(
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
        return self.root_dir / "archives"

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
        return utils.ensure_dir(path) if create else path

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
        return utils.ensure_dir(path) if create else path

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

        df.insert(loc=0, column="uuid", value=None)
        self._con.con.register("metadata_tile", df)

        # TODO: consider configurable duplicate behaviour (REPLACE or IGNORE)
        self._con.raw_sql(f"INSERT OR IGNORE INTO {table} FROM metadata_tile")

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
    def get_mapillary_download_records(self) -> list[tuple[str, str]]:
        """Return list of (id, url, location) for Mapillary images to download."""

        keys = {
            "uid": "uuid",
            "id": "id",
            "url": "thumb_2048_url",
            "shard": "shard",
            "location": "geometry",
        }
        t_map = self._con.table("mapillary")
        t_img = self._con.table("images")
        t = t_map.outer_join(t_img, t_map.uuid == t_img.uuid)
        t = t.select(**keys)
        data = t.to_pyarrow().to_pydict()
        if len(data) == 0:
            return [() * len(keys)]
        return list(zip(*[data[k] for k in keys]))

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
        INSERT INTO images (id, source, geometry)
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
            "collection": collections,
            "model": models,
            "run": runs,
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

        t_im = self._con.table("images")
        t_col = self._con.table("collections")
        t_seg = self._con.table("segmentations")

        col_filtered = t_col.filter([t_col.name == collection])
        image_count = int(col_filtered.select("name").count().to_pyarrow())
        if image_count == 0:
            logger.error(
                f"Collection '{collection}' does not exist in the current project."
            )
            return

        seg_filtered = col_filtered.outer_join(
            t_seg.filter(
                [
                    t_seg.collection == collection,
                    t_seg.model == model,
                    t_seg.run == run,
                ]
            ),
            t_col.name == t_seg.collection,
        )

        archive = seg_filtered.select("archive").to_pyarrow().to_pydict()["archive"]
        if len(archive) > 0:
            archive = archive[0]
        else:
            archive = utils.uuid7()
            self.add_segmentations(collection, model, run, archive)

        archive_path = utils.ensure_dir(self.get_archive_path(model) / str(archive))
        existing = set([p.stem for p in archive_path.iterdir()])

        keys = ("hash", "path", "source")
        t_seg_flt = seg_filtered.inner_join(t_im, seg_filtered.hash == t_im.hash)
        t_all = t_seg_flt.outer_join(t_seg, t_seg_flt.name == t_seg.collection)
        entries = t_all.select(*keys).to_pyarrow().to_pydict()

        processed = {}
        unprocessed = {}
        for h, p, s in zip(entries["hash"], entries["path"], entries["source"]):
            if h is not None and h.hex() in existing:
                processed.add(h)
            else:
                unprocessed[h] = self.get_image_path(s) / p

        return processed, unprocessed

    def register_images(
        self,
        data: ibis.Table,
    ):
        """
        Register downloaded (local) images with the database.

        Args:
            data: A temporary Ibis table.
        """

        if "images" not in self._con.tables:
            logger.error(
                f"Project database seems to be corrupted: missing table 'images'."
            )
            return

        try:
            self._con.con.register("new_images", data.to_pandas())
            self._con.raw_sql(f"INSERT OR REPLACE INTO images FROM new_images;")

        except duckdb.ConstraintException as e:
            logger.debug(f"Error updating table 'images'")

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
