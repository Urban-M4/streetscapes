import shutil
from typing import Any
import uuid
from pathlib import Path

import duckdb
import ibis
import orjson as oj
import pandas as pd
import shapely as shp
from pandas import DataFrame
from shapely.geometry import box

from streetscapes import CFG, logger, utils
from streetscapes.utils.bbox import Bbox

from rich.progress import track


def _format_image(
    uid: uuid.UUID,
    source: str,
    shard: str | None,
    notes: str = "",
    tags: str | list[str] | None = None,
    rating: int = 0,
) -> dict[str, Any]:
    """
    Prepare image metadata for entry into for the database.

    Args:
        uid: UUID of the image.
            Should be generated automatically from the SHA-256 hash of the image.
            See `utils.hash2uuid()`.
        source: Image provenance.
        shard: Shard where the image is located
        notes: Freestyle notes.
        tags: List of image tags descriptive of the image.
        rating: Image quality rating.
        overwrite: Replace or ignore conflicting data.
    """

    if isinstance(tags, str):
        tags = [tags]

    return {
        "uuid": ibis.uuid(uid).to_pyarrow(),
        "source": source,
        "shard": shard,
        "notes": notes,
        "tags": tags if tags is not None else [],
        "rating": rating,
    }


class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    # Core tables that need to be present in the project database.
    core_tables = {
        "images": {
            "schema": {
                "uuid": "UUID PRIMARY KEY",
                "source": "STRING NOT NULL",
                "shard": "STRING",
                "notes": "STRING",
                "tags": "STRING[] NOT NULL",
                "rating": "INTEGER NOT NULL DEFAULT 0",  # 0-5
            },
            "init": [],
        },
        "runs": {
            "schema": {
                "run": "STRING PRIMARY KEY",
                "timestamp": "TIMESTAMP NOT NULL",
                "model": "STRING",
                "metadata": "JSON",
            },
            "init": [],
        },
        "segmentations": {
            "schema": {
                "run": "STRING NOT NULL",
                "curated": "BOOL NOT NULL DEFAULT FALSE",
                "image": "UUID NOT NULL",
                "labels": "STRING[]",
                "rating": "INTEGER",  # 0-5
                "polygons": "GEOMETRY",
            },
            "init": [
                "ALTER TABLE segmentations ADD PRIMARY KEY (run, curated, image)"
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
        "local": {
            "schema": {
                "image": "UUID PRIMARY KEY",
                "make": "STRING",
                "model": "STRING",
                "orientation": "UBIGINT",
                "timestamp": "TIMESTAMP",
                "width": "UBIGINT",
                "height": "UBIGINT",
                "altitude": "FLOAT4",
                "geometry": "GEOMETRY",
                "is_pano": "BOOL",
                "iso": "UBIGINT",
                "focal_length": "FLOAT4",
                "exposure": "FLOAT4",
                "fstop": "FLOAT4",
            },
            "init": [],
        },
        # TODO: KartaView and Amsterdam tables
    }

    def __init__(
        self,
        name: str | None = None,
        image_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
    ):

        self.name = name or CFG.active_project

        # Directory for projects (databases + segmentations)
        self.project_dir = Path(project_dir or CFG.project_dir)

        # Directory for cached data (images)
        self.image_dir = Path(image_dir or CFG.image_dir)

        CFG.active_project = self.name
        CFG.save()

        # Internal attributes
        # ==================================================
        self._timestamp_resolution = "milliseconds"

        # Database connection
        self._con = ibis.duckdb.connect(
            self.db_path,
            extensions=["spatial", "json"],
        )
        self.bootstrap()

    @property
    def db_path(self) -> Path:
        return self.project_dir / f"{self.name}.duckdb"

    @property
    def archive_path(self) -> Path:
        return self.project_dir / "archives"

    @property
    def image_path(self) -> Path:
        return self.image_dir / "images"

    @property
    def tables(self) -> list[str]:
        return self._con.tables  # type: ignore[no-any-return]

    def table(
        self,
        name: str,
    ) -> ibis.Table:
        """
        Return an Ibis table for the requested name.

        Args:
            name: Table name.

        Returns:
            An optional Ibis table.
        """
        if name in self._con.tables:
            return self._con.table(name)
        msg = f"Table '{name}' not found in tables: {self._con.tables}"
        raise TableNotFoundError(msg)

    def schema(
        self,
        name: str,
    ) -> dict:
        """Return the schema for a table.

        Args:
            name: Table name.

        Returns:
            An optional table schema.
        """
        if name in self.core_tables:
            return self.core_tables[name]["schema"]  # type: ignore[return-value]
        msg = f"Table schema '{name}' not found in core tables: {self.core_tables}"
        raise TableNotFoundError(msg)

    def bootstrap(
        self,
        overwrite: bool = False,
    ) -> None:
        """Bootstrap the project with the core tables.

        Tables are specified in the `core_tables` attribute.

        Args:
            overwrite: Overwrite an existing database.
        """
        # Tables that should exist in every project.
        # Just update the set with table names
        # and define the schema in the `schema` property.
        for name, _ in self.core_tables.items():
            self.ensure_table(name, overwrite=overwrite)

    def ensure_table(
        self,
        name: str,
        schema: dict | ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> None:
        """Ensure that a table exists with the given schema.

        Args:
            name: Table name.
            schema: Schema to use for the table if it doesn't exist.
            overwrite: Overwrite the table if it exists.
        """
        if name in self.tables and not overwrite:
            return

        table = self.core_tables.get(name)
        if table is None:
            raise ValueError(f"Invalid table '{name}'.")

        schema = schema or self.schema(name)
        if schema is None:
            raise ValueError(f"Please provide a valid schema for table '{name}'.")

        if overwrite:
            sql = f"CREATE OR REPLACE TABLE {name}"
        else:
            sql = f"CREATE TABLE IF NOT EXISTS {name}"

        fields = ", ".join(
            [f"{fname} {definition}" for fname, definition in schema.items()]  # type: ignore[union-attr]
        )
        sql = f"{sql} ({fields})"

        self._con.raw_sql(sql)

        # Run init clauses, if any
        inits = table.get("init", [])
        for sql in inits:
            self._con.raw_sql(sql)

    def get_image_dir_for_source(
        self,
        source: str | None = None,
        create: bool = False,
    ) -> Path:
        """
        Get the path to the directory where downloaded images
        are stored, optionally specifying a source.

        Args:
            source: The source name (e.g., 'mapillary').
            create: Optionally create the directory.

        Returns:
            A Path object.
        """
        path = self.image_path
        if source is None:
            source = CFG.local_cache_dir_name
        path /= source
        return utils.ensure_dir(path) if create else path

    def get_image_paths_from_uuids(
        self,
        uids: uuid.UUID | list[uuid.UUID],
    ) -> dict[uuid.UUID, tuple[Path, str]]:
        """
        Get image paths from UUIDs.

        Args:
            uids: Image UUID(s) (produced with SHA-256, see utils.hash2uuid)

        Returns:
            The paths to the images.
        """

        if isinstance(uids, uuid.UUID | str):
            uids = [uids]

        uids = [uuid.UUID(u) if isinstance(u, str) else u for u in uids]

        t = self.table("images")
        results = (
            t.filter([t.uuid.isin(uids)])
            .select(["uuid", "source", "shard"])
            .to_pyarrow()
            .to_pylist()
        )

        paths: dict[uuid.UUID, tuple[Path, str]] = {}
        for result in results:
            uid, source, shard = (
                uuid.UUID(result["uuid"]),
                str(result["source"]),
                result["shard"],
            )

            src_dir = self.get_image_dir_for_source(source)
            if shard is not None:
                src_dir /= shard

            fpath = list(src_dir.glob(f"*{uid}*"))
            if len(fpath) > 0:
                if fpath[0].is_file():
                    paths[uid] = fpath[0], source

        return paths

    def get_archive_dir_for_model(
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
            t = self.table("segmentations")
            t_run = self.table("runs")
            t = t.inner_join(t_run, t.run == t_run.run)
            if curated is not None:
                t = t.filter([t.curated == curated])

        return t.to_pyarrow().to_pylist()  # type: ignore[no-any-return]

    def add_run(
        self,
        run: str | None = None,
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
        ts = utils.iso_timestamp(self._timestamp_resolution, utc=False)

        if run is None:
            run = f"{model or 'unknown'}-{ts}"

        data = {
            "run": [run],
            "timestamp": [ts],
            "model": [model],
            "metadata": [oj.dumps(metadata)],
        }

        return self.update_table("runs", data, overwrite)  # type: ignore[no-any-return]

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
        ts = utils.iso_timestamp(self._timestamp_resolution, utc=False)
        data: dict[str, list] = {k: [] for k in self.schema("runs")}

        for r in runs:
            model = r.get("model", "unknown")
            r.setdefault("run", f"{model}-{ts}")
            r.setdefault("timestamp", ts)
            r["metadata"] = oj.dumps(r.get("metadata"))
            for k in data:
                data[k].append(r.get(k))

        return self.update_table("runs", data, overwrite)  # type: ignore[no-any-return]

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
        if t is None:
            raise ValueError

        t = t.filter([t.run == run, t.curated == curated, t.image == image])

        result = t.to_pyarrow().to_pylist()

        if len(result) > 0:
            return result[0]  # type: ignore[no-any-return]
        return []

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

        return t.to_pyarrow().to_pylist()  # type: ignore[no-any-return]

    def add_segmentation(
        self,
        run: str,
        image: uuid.UUID | str,
        labels: list[str],
        curated: bool = False,
        rating: int = 0,
        polygons: shp.GeometryCollection | None = None,
        overwrite: bool = False,
    ):
        """Add a new segmentation to the database.

        Args:
            run: Model run ID.
            image: Image ID.
            labels: A list of labels.
            curated: Curation status.
            rating: Segmentation rating.
            polygons: A Shapely GeometryCollection.
            overwrite: Replace or ignore conflicting data.
        """

        data = {
            "run": [run],
            "curated": [curated],
            "image": [ibis.uuid(image).to_pyarrow()],
            "labels": [labels],
            "rating": [rating],
            "polygons": [polygons],
        }

        result = self.update_table("segmentations", data, overwrite=True)

        return result

    def add_segmentations(
        self,
        segmentations: list[dict],
        overwrite: bool = False,
    ):
        """
        Add a new set of segmentations to the database.

        Args:
            segmentations: A list of dictionaries containing segmentation data.
            overwrite: Replace or ignore conflicting data.
        """
        data: dict[str, list] = {k: [] for k in self.schema("segmentations")}

        for s in segmentations:
            s["image"] = ibis.uuid(s["image"]).to_pyarrow()
            s.setdefault("curated", False)
            for k in data:
                data[k].append(s.get(k))

        return self.update_table("segmentations", data, overwrite)

    def get_segmentation_status(
        self,
        uids: list[uuid.UUID] | list[str],
        run: str,
    ) -> tuple[set[uuid.UUID], dict[uuid.UUID, tuple[Path, str]]]:
        """
        Filter out processed images.

        Args:
            images: Image UUIDs.
            run: Query the status for a specific run ID.

        Returns:
            Sets of UUIDs for processed and unprocessed images.
        """
        uids = [uuid.UUID(u) if isinstance(u, str) else u for u in uids]

        t_seg = self.table("segmentations")
        t_seg_flt = t_seg.filter([t_seg.image.isin(uids), t_seg.run == run]).select(
            "image"
        )

        processed = set(t_seg_flt.to_pyarrow().to_pydict()["image"])
        missing = list(set(uids).difference(processed))
        unprocessed = self.get_image_paths_from_uuids(missing)
        return processed, unprocessed

    def add_images(
        self,
        images: list[dict],
        exif_data: list[dict] | None = None,
        overwrite: bool = False,
    ):
        """
        Register downloaded (local) images into the database.

        Args:
            images: A list of dictionaries containing image information.
            exif_data: Metadata extracted from the images' EXIF tags. Note: only used for locally imported images.
            overwrite: Replace or ignore conflicting data.
        """

        # Entries for the `images` table.
        img_data: dict[str, list] = {column: [] for column in self.schema("images")}

        for image in images:
            for column in img_data:
                img_data[column].append(image.get(column))
        self.update_table("images", img_data, overwrite)

        # Entries for the `local` table.
        if exif_data is not None:
            metadata: dict[str, list] = {column: [] for column in self.schema("local")}
            for exif in exif_data:
                for column in metadata:
                    metadata[column].append(exif.get(column))
            self.update_table("local", metadata, overwrite)

        return img_data

    def add_local_images(
        self,
        path: Path | str,
        shard: str | None = None,
        overwrite: bool = False,
    ):
        """
        Add images from a directory.

        Args:
            path: A directory containing images.
            shard: An optional shard (=subdirectory) to use.
            overwrite: Overwrite existing entries.
        """

        path = Path(path)
        image_paths = utils.get_image_paths(path)
        image_data = []
        exif_data = []
        image_dir = self.get_image_dir_for_source(CFG.local_cache_dir_name, create=True)
        if shard is not None:
            image_dir = utils.ensure_dir(image_dir / shard)

        ti = self.table('images')

        # Iterate over all images and copy them to the local directory.
        # Any EXIF metadata in the images themselves can be used to
        # populate the `local` table.
        for ip in track(image_paths, description="Adding local images..."):
            uid = utils.get_image_uuid(ip)
            if (ti.filter(ti.uuid == uid).count().to_pandas() > 0) and not overwrite:
                continue

            new_fname = f"{uid}{ip.suffix}".lower()
            new_fpath = image_dir / new_fname

            if not new_fpath.exists() or overwrite:
                shutil.copy2(ip, new_fpath)

            #
            entry = _format_image(
                uid,
                source="local",
                shard=shard,
            )
            image_data.append(entry)
            exif = utils.extract_exif_data(ip)
            exif["image"] = uid
            exif_data.append(exif)

        return self.add_images(image_data, exif_data, overwrite)

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
        # TODO do not depend of order in df columns, but used named columns in SQL query
        self._con.raw_sql(f"INSERT OR IGNORE INTO {table} FROM metadata_tile")

    def ingest_image_records(self, records: list[dict]):
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

    def get_image_uuids(self) -> list[uuid.UUID]:
        """Return a list of all image UUIDs in the project."""
        t = self.table("images")
        if t is None:
            raise ValueError("The 'images' table is not present in the database.")
        result = t.select("uuid").to_pyarrow().to_pydict()
        return [uuid.UUID(u) for u in result["uuid"]]

    def filter_bbox(
        self,
        table: str,
        bbox: Bbox,
    ) -> ibis.Table:
        """
        Return an Ibis table expression filtered by a bounding box.

        Args:
            table: The table name.
            bbox: The bounding box.

        Returns:
            An Ibis table.
        """

        t = self.table(table)
        envelope_expr = ibis.literal(box(*bbox).wkt, type="geospatial:geometry")
        return t.filter(t.geometry.within(envelope_expr))

    def update_table(
        self,
        table: str,
        data: dict,
        overwrite: bool = False,
    ):
        """Update a table.

        Args:
            table: The table to update.
            data: Updated data.
            overwrite: Replace or ignore conflicting data.
        """
        if table not in self.tables:
            logger.error(
                f"Project database seems to be corrupted: missing table '{table}'."
            )
            return None

        try:
            alt = "REPLACE" if overwrite else "IGNORE"
            self._con.con.register("updated_df", pd.DataFrame(data))
            if table == "segmentations":
                self._con.raw_sql(f"INSERT INTO {table} FROM updated_df;")
            else:
                self._con.raw_sql(f"INSERT OR {alt} INTO {table} FROM updated_df;")

            return data

        except duckdb.ConstraintException as e:
            logger.error(f"Constraint violation on '{table}': {e}")

        except Exception as e:
            logger.error(f"Error updating table '{table}': {e}")

    # TODO: could generalize to "get_records(table, columns, include='missing')"
    def get_mapillary_download_records(self) -> list[tuple[Any, ...]]:
        """Return list of (id, url, location) for Mapillary images to download."""
        keys = {
            "image": "image",
            "id": "id",
            "url": "thumb_2048_url",
            "shard": "shard",
            "location": "geometry",
            "is_pano": "is_pano",
            "camera_type": "camera_type",
        }
        t_map = self.table("mapillary")
        t_img = self.table("images")
        if t_img is None or t_map is None:
            raise ValueError(
                "Required tables 'mapillary' and 'images' are not present in the database."
            )
        t = t_map.outer_join(t_img, t_map.image == t_img.uuid)
        t = t.select(**keys)
        # images that have been downloaded have `image IS NOT NULL`,
        # so we filter those out to get the missing ones.
        data = t.filter([t.image.isnull(), t.url.notnull()]).to_pyarrow().to_pydict()
        if len(data) == 0:
            return [() * len(keys)]
        return list(zip(*[data[k] for k in keys]))

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


class TableNotFoundError(Exception):
    pass
