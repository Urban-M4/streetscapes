from pathlib import Path

import ibis
from pandas import DataFrame
from shapely.geometry import box
import numpy as np
import uuid
import duckdb
import platformdirs as pdirs
import orjson as oj
import shapely as shp

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
        "runs": {
            "schema": {
                "run": "STRING NOT NULL PRIMARY KEY",
                "timestamp": "TIMESTAMP NOT NULL",
                "model": "STRING",
                "metadata": "JSON",
            },
            "init": [],
        },
        "segmentations": {
            "schema": {
                "run": "STRING NOT NULL",
                "curated": "BOOL NOT NULL DEFAULT TRUE",
                "image": "UUID NOT NULL",
                "labels": "STRING[] NOT NULL",
                "polygons": "GEOMETRY",
            },
            "init": [
                "ALTER TABLE segmentations ADD PRIMARY KEY (run, curated, image);"
            ],
        },
        "mapillary": {
            "schema": {
                "image": "UUID",
                "altitude": "FLOAT8",
                "atomic_scale": "FLOAT8",
                "camera_type": "STRING",
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
        # Timestamp precision
        self._timespec = "microseconds"
        self._db_name = "metadata"
        self._con = ibis.duckdb.connect(
            self.db_path,
            extensions=["spatial", "json"],
        )

        if self._db_name not in self._con.list_databases():
            self.bootstrap()

        self._use_db()

    @property
    def db_path(self) -> Path:
        return self.project_home / f"{self.name}.duckdb"

    @property
    def archive_path(self) -> Path:
        return self.root_dir / "archives"

    @property
    def image_path(self) -> Path:
        return self.data_home / "images"

    @property
    def tables(self) -> list[str]:
        return self._con.tables

    def table(
        self,
        name: str,
    ) -> ibis.Table | None:
        """
        Return an Ibis table for the requested name.

        Args:
            name: Table name.

        Returns:
            An optional Ibis table.
        """
        if name in self._con.tables:
            return self._con.table(name)

    def _use_db(
        self,
        db: str | None = None,
    ):
        self._con.raw_sql(f"USE {db or self._db_name};")

    def bootstrap(
        self,
        overwrite: bool = True,
    ):
        """
        Bootstrap the project with the core tables
        specified in the `core_tables`.

        Args:
            overwrite: Overwrite an existing database.
        """

        if self._db_name in self._con.list_databases():
            if overwrite:
                # First, drop all the tables:
                for t in self.tables:
                    self._con.drop_table(t, database=self._db_name)
                self._con.drop_database(self._db_name, force=True)
            else:
                return

        self._con.create_database(self._db_name)
        self._use_db()

        # Tables that should exist in every project.
        # Just update the set with table names
        # and define the schema in the `schema` property.
        for name, items in self.core_tables.items():
            self.ensure_table(name, overwrite=overwrite)

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
        if name in self.tables and not overwrite:
            return self.table(name)

        if schema is None:
            table = self.core_tables.get(name)
            if table is None or (schema := table.get("schema")) is None:
                raise ValueError(f"Please provide a valid schema for table '{name}'.")

        if overwrite:
            sql = f"CREATE OR REPLACE TABLE {name}"
        else:
            sql = f"CREATE TABLE IF NOT EXISTS {name}"

        fields = ", ".join(
            [f"{fname} {definition}" for fname, definition in schema.items()]
        )
        sql = f"{sql} ({fields});"

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

    def get_image_path_from_uuid(
        self,
        uid: uuid.UUID,
    ) -> Path:
        """
        Get the path to an image from its UUID.

        Args:
            uid: Image UUID (produced with SHA-256, see utils.hash2uuid)

        Returns:
            The path to the image.
        """

        t = self.table("images")
        t = t.filter([t.uuid == uid]).to_pyarrow().to_pylist()

        if len(t) == 0:
            return

        t = t[0]
        source, shard = t["source"], t["shard"]
        idir = self.get_image_path(source) / shard
        files = idir.glob(f"{uid}")
        if len(img_file) > 0:
            img_file = files[0]
            if img_file.is_file():
                return img_file

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

    def get_run(
        self,
        result: uuid.UUID,
        segmentations: bool = False,
        curated: bool | None = None,
    ) -> list[dict]:
        """
        Get (an optionally curated) segmentation run.

        Args:
            run: The model run.
            segmentations: If True, get the associated segmentations as well.
            curated: Optionally filter segmentations by 'curated' status.

        Returns:
            UUID of the archive.
        """

        t = self.table("runs")
        t = t.filter([t.run == result])

        if segmentations:
            t_seg = self.table("segmentations")
            t = t.inner_join(t_seg, t_seg.run == t.run)
            if curated is not None:
                t = t.filter([t.curated == curated])

        result = t.to_pyarrow().to_pylist()

        return result

    def add_run(
        self,
        run: uuid.UUID | None = None,
        model: str | None = None,
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> dict:
        """
        Add a run for a model and its associated metadata.

        Args:
            run: The model run ID (optional, UUID7 used by default).
            model: The model used for this run.
            metadata: Any metadata pertaining to the model or the run.
            overwrite: Overwrite an existing entry.

        Returns:
            The data added to the database.
        """
        if run is None:
            run = utils.uuid7(True)

        data = {
            "run": [run],
            "timestamp": [utils.iso_timestamp(self._timespec)],
            "model": [model],
            "metadata": [oj.dumps(metadata)],
        }

        result = self.update_table("runs", data, overwrite)

        return result

    def add_runs(
        self,
        runs: list[dict],
        overwrite: bool = False,
    ) -> dict:
        """
        Add a run for a model and its associated metadata.

        Args:
            runs: List of run data as dictionaries.
            overwrite: Overwrite an existing entry.

        Returns:
            The data added to the database.
        """

        data = {k: [] for k in self.core_tables["runs"]["schema"]}

        for r in runs:
            r.setdefault("run", utils.uuid7(True))
            r.setdefault("timestamp", utils.iso_timestamp(self._timespec))
            r["metadata"] = oj.dumps(r.get("metadata"))
            for k in data:
                data[k].append(r.get(k))

        result = self.update_table("runs", data, overwrite)

        return result

    def get_segmentation(
        self,
        run: str,
        curated: bool,
        image: uuid.UUID | str,
    ) -> list[dict]:
        """
        Get a segmentation by UUID + curation status.

        Args:
            run: Run ID.
            curated: Curation status.
            image: Image ID.

        Returns:
            Segmentation instance.
        """

        t = self.table("segmentations")
        t = t.filter([t.run == run, t.curated == curated, t.image == image])

        result = t.to_pyarrow().to_pylist()

        if len(result) > 0:
            return result[0]

    def get_segmentations(
        self,
        image: uuid.UUID | str,
        runs: str | list[str] | None = None,
        curated: bool | None = None,
    ) -> list[dict]:
        """
        Get all segmentations of an image,
        optionally filtered by run ID and curation status.

        Args:
            image: Image ID.
            runs: Run IDs.
            curated: Curation status.

        Returns:
            A list of segmentation instances.
        """

        t = self.table("segmentations")
        flt = [t.image == image]
        if runs is not None:
            # Filter by run ID.

            if isinstance(runs, str):
                flt.append(t.run == runs)

            elif isinstance(runs, (list, tuple, set)):
                flt.append(t.run.isin(list(map(str, runs))))

        if curated is not None:
            # Filter by curation status.
            flt.append(t.curated == curated)

        t = t.filter(flt)

        result = t.to_pyarrow().to_pylist()

        return result

    def add_segmentation(
        self,
        curated: bool,
        image: uuid.UUID | str,
        labels: list[str],
        polygons: shp.GeometryCollection,
        run: str | None = None,
        model: str | None = None,
        metadata: dict | None = None,
        overwrite: bool = True,
    ):
        """
        Add a new segmentation to the database.

        Args:
            curated: Curation status.
            image: Image ID.
            labels: A list of labels.
            polygons: A Shapely GeometryCollection.
            run: Optional run ID.
            model: The model used for the segmentation.
            metadata: Any metadata associated with this segmentation.
            overwrite: Replace or ignore conflicting data.
        """

        if run is None:
            run = utils.uuid7(True)

        # Store the run
        self.add_run(run, model, metadata, overwrite)

        data = {
            "run": run,
            "curated": curated,
            "image": ibis.uuid(image).to_pyarrow(),
            "labels": labels,
            "polygons": polygons,
        }

        result = self.update_table("segmentations", data, overwrite)

        return result

    def add_segmentations(
        self,
        segmentations: list[dict],
        run: str | None = None,
        model: str | None = None,
        metadata: dict | None = None,
        overwrite: bool = True,
    ):
        """
        Add a new segmentation to the database.

        Args:
            segmentations: A list of dictionaries containing segmentation data.
            run: Optional run ID.
            model: The model used for the segmentation.
            metadata: Any metadata associated with this segmentation.
            overwrite: Replace or ignore conflicting data.
        """

        if run is None:
            run = utils.uuid7(True)

        # Store the run
        self.add_run(run, model, metadata, overwrite)

        data = {k: [] for k in self.core_tables["segmentations"]["schema"]}

        for s in segmentations:
            s["image"] = ibis.uuid(s["image"]).to_pyarrow()
            for k in data:
                data[k].append(s.get(k))

        return self.update_table("segmentations", data, overwrite)

    def get_segmentation_status(
        self,
        images: uuid.UUID,
        run: str,
    ) -> tuple[set[uuid.UUID], ...]:
        """
        Filter out processed images.

        Args:
            images: Image UUIDs.
            run: Query the status for a specific run ID.

        Returns:
            Sets of UUIDs for processed and unprocessed images.
        """

        images = list(map(uuid.UUID, images))

        t_seg = self.table("segmentations")
        t_seg_flt = t_seg.filter([t_seg.image.isin(images), t_seg.run == run]).select(
            "image"
        )

        processed = set(t_seg_flt.to_pyarrow().to_pydict()["image"])
        unprocessed = set(images).difference(processed)

        return processed, unprocessed

    def add_image(
        self,
        uid: uuid.UUID,
        source: str | None = None,
        shard: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        rating: int | None = None,
        overwrite: bool = True,
    ):
        """
        Register downloaded (local) images into the database.

        Args:
            uid: UUID of the image.
                Should be generated automatically from the SHA-256 hash of the image.
                See `utils.hash2uuid()`.
            source: Image provenance.
            shard: Shard segment of the image's storage path.
            notes: Freestyle notes.
            tags: List of image tags descriptive of the image.
            rating: Image quality rating.
            overwrite: Replace or ignore conflicting data.
        """

        data = {
            "uuid": [ibis.uuid(uid).to_pyarrow()],
            "source": [source],
            "shard": [shard],
            "notes": [notes],
            "tags": [tags],
            "rating": [rating],
        }

        return self.update_table("images", data, overwrite)

    def add_images(
        self,
        data: dict,
        overwrite: bool = True,
    ):
        """
        Register downloaded (local) images into the database.

        Args:
            data: A dictionary of image information.
            overwrite: Replace or ignore conflicting data.
        """

        _data = {
            "uuid": [],
            "source": [],
            "shard": [],
            "notes": [],
            "tags": [],
            "rating": [],
        }

        for i in data:
            if "uuid" not in i:
                raise KeyError(f"The 'uuid' field is mandatory.")
            i["uuid"] = ibis.uuid(i["uuid"]).to_pyarrow()

            for k in _data:
                _data[k].append(i.get(k))

        return self.update_table("images", _data, overwrite)

    def ingest_mapillary(self, df: DataFrame, table: str = "mapillary"):
        """Ingest a DataFrame of Mapillary metadata."""

        # Ensure that that the camera_parameters column is a list of floats
        df["camera_parameters"] = df["camera_parameters"].apply(
            lambda params: (
                [float(params)]
                if isinstance(params, int | float)
                else list(map(float, params))
            )
        )

        df.insert(loc=0, column="uuid", value=None)

        self._con.con.register("metadata_tile", df)

        # TODO: consider configurable duplicate behaviour (REPLACE or IGNORE)
        self._con.raw_sql(f"INSERT OR IGNORE INTO {table} FROM metadata_tile")

    def filter_bbox(self, table: str, bbox: Bbox):
        """Return an Ibis table expression filtered by a bounding box."""

        table = self.ensure_table(table)
        envelope_expr = ibis.literal(box(*bbox).wkt, type="geospatial:geometry")
        return table.filter(table.geometry.within(envelope_expr))

    def register_collection(
        self,
        collection: str,
        uids: list[uuid.UUID],
        overwrite: bool = True,
    ):
        """
        Register a downloaded (local) image into the database.

        Args:
            collection: The collection to add the images to.
            uids: Image UUIDs.
            overwrite: Overwrite existing entries with new ones.
        """

        if overwrite:
            sql = f"DELETE FROM collections WHERE name='{collection}';"
            self._con.raw_sql(sql)

        uids = [ibis.uuid(u).to_pyarrow() for u in uids]
        data = {
            "name": [collection for _ in range(len(uids))],
            "image": uids,
        }

        self.update_table("collections", data, overwrite)

    def save_segmentation(
        self,
        path: Path | str,
        instances: np.ndarray,
        fmt: str = "npz",
    ):
        """
        Save a segmentation.

        Args:
            path: The file to save instances to.
            instances: NumPy array of instance masks.
            fmt: Format of the saved file.
        """

        match fmt:
            case "npz":
                np.savez_compressed(path, instances)
            case "npy":
                np.save(path, instances)
            # TODO: Add parquet and efficient geometry storage.
            # NOTE: Check if it's possible do do away with this step
            # entirely by storing segmentation outlines straight into the database.
            case _:
                np.savez_compressed(path, instances)

    def update_table(
        self,
        table: str,
        data: dict,
        overwrite: bool = True,
    ):
        """
        Update a table.

        Args:
            table: The table to update.
            data: Updated data.
            overwrite: Replace or ignore conflicting data.
        """

        if table not in self.tables:
            logger.error(
                f"Project database seems to be corrupted: missing table '{table}'."
            )
            return

        mt = ibis.memtable(data)

        try:
            alt = "REPLACE" if overwrite else "IGNORE"
            self._con.con.register("updated_df", mt.to_pandas())
            self._con.raw_sql(f"INSERT OR {alt} INTO {table} FROM updated_df;")
            result = mt.to_pyarrow().to_pylist()
            return result

        except duckdb.ConstraintException as e:
            logger.debug(f"Constraint violation on '{table}': {e}")

        except Exception as e:
            logger.debug(f"Error updating table '{table}': {e}")

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
        t_map = self.table("mapillary")
        t_img = self.table("images")
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

    # Export functions
    def _select_nonspatial(self, table: str) -> str:
        """Return SELECT clause converting geometry to WKT if present."""
        if "geometry" in self.table(table).columns:
            return "* EXCLUDE (geometry), ST_AsText(geometry) AS geometry"
        return "*"

    def _require_geometry(self, table: str, fmt: str):
        """Ensure table has a geometry column before geospatial export."""
        if "geometry" not in self.table(table).columns:
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
        self._con.raw_sql(f"""COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}'
                (HEADER, DELIMITER ',');""")

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
