from pathlib import Path

import ibis
from shapely.geometry import box

from streetscapes import config


class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    def __init__(self, name: str = "streetscapes"):
        # TODO: also read name from config? But keep option to overwrite?
        self.name = name
        self.data_home = Path(config.get("data_home"))

        database_path = self.data_home / "projects" / f"{name}.duckdb"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = ibis.duckdb.connect(database_path)
        self.con.raw_sql("INSTALL spatial; LOAD spatial;")

    def image_dir(self, source: str | None = None):
        if source is None:
            return self.data_home / "images"
        return self.data_home / "images" / source

    def get_table(self, name: str):
        """Return an ibis table reference."""
        return self.con.table(name)

    def ingest_mapillary(self, df, table_name="mapillary"):
        """Ingest a DataFrame of Mapillary metadata."""
        self.con.con.register("metadata_tile", df)
        if table_name not in self.con.list_tables():
            self.con.raw_sql(f"""
                CREATE TABLE {table_name} AS
                SELECT
                    * EXCLUDE (geometry),
                    ST_GeomFromText(geometry) AS geometry,
                FROM metadata_tile;
                ALTER TABLE {table_name} ADD PRIMARY KEY (id);
            """)
        else:
            # TODO: consider configurable duplicate behaviour (REPLACE or IGNORE)
            self.con.raw_sql(f"""
                INSERT OR REPLACE INTO {table_name}
                SELECT
                    * EXCLUDE (geometry),
                    ST_GeomFromText(geometry) AS geometry,
                FROM metadata_tile
            """)

    def filter_bbox(self, table_name, bbox):
        """Return an Ibis table expression filtered by a bounding box."""

        table = self.get_table(table_name)
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
        self.con.raw_sql("""
            CREATE TABLE IF NOT EXISTS local_images (
                uuid TEXT PRIMARY KEY,
                id TEXT NOT NULL,
                source TEXT NOT NULL,
                path TEXT NOT NULL,
                geometry GEOMETRY,
            )
        """)

        sql = """
        INSERT INTO local_images (uuid, id, source, path, geometry)
        VALUES (GEN_RANDOM_UUID(), ?, ?, ?, ST_GeomFromWKB(?))
        ON CONFLICT DO NOTHING
        """

        params = [
            (r["id"], r["source"], str(r["path"]), r["geometry"]) for r in records
        ]

        # Execute as batch
        self.con.con.executemany(sql, params)
