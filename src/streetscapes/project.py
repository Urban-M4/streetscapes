import ibis
from shapely.geometry import box


class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    def __init__(self, db_path: str = "streetscapes.duckdb"):
        self.db = ibis.duckdb.connect(db_path)
        self.db.raw_sql("INSTALL spatial; LOAD spatial;")

    def get_table(self, name: str):
        """Return an ibis table reference."""
        return self.db.table(name)

    def ingest_mapillary(self, df, table_name="mapillary"):
        """Ingest a DataFrame of Mapillary metadata."""
        self.db.con.register("metadata_tile", df)
        if table_name not in self.db.list_tables():
            self.db.raw_sql(f"""
                CREATE TABLE {table_name} AS
                SELECT
                    * EXCLUDE (geometry, computed_geometry),
                    ST_GeomFromText(geometry) AS geometry,
                    ST_GeomFromText(computed_geometry) AS computed_geometry
                FROM metadata_tile;
                ALTER TABLE {table_name} ADD PRIMARY KEY (id);
            """)
        else:
            # TODO: consider configurable duplicate behaviour (REPLACE or IGNORE)
            self.db.raw_sql(f"""
                INSERT OR REPLACE INTO {table_name}
                SELECT
                    * EXCLUDE (geometry, computed_geometry),
                    ST_GeomFromText(geometry) AS geometry,
                    ST_GeomFromText(computed_geometry) AS computed_geometry
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
        if "geometry" in self.db.table(table).columns:
            return "* EXCLUDE (geometry), ST_AsText(geometry) AS geometry"
        return "*"

    def _require_geometry(self, table: str, fmt: str):
        """Ensure table has a geometry column before geospatial export."""
        if "geometry" not in self.db.table(table).columns:
            raise ValueError(f"{fmt} export requires a 'geometry' column in '{table}'.")

    def export_parquet(self, table: str, output_path: str):
        self.db.raw_sql(f"COPY {table} TO '{output_path}' (FORMAT PARQUET);")

    def export_json(self, table: str, output_path: str):
        nonspatial = self._select_nonspatial(table)
        self.db.raw_sql(
            f"COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}' (FORMAT JSON);"
        )

    def export_csv(self, table: str, output_path: str):
        nonspatial = self._select_nonspatial(table)
        self.db.raw_sql(
            f"""COPY (SELECT {nonspatial} FROM {table}) TO '{output_path}'
                (HEADER, DELIMITER ',');"""
        )

    def export_gpkg(self, table: str, output_path: str):
        self._require_geometry(table, "GeoPackage")
        self.db.raw_sql(
            f"COPY {table} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GPKG');"
        )

    def export_geojson(self, table: str, output_path: str):
        self._require_geometry(table, "GeoJSON")
        self.db.raw_sql(
            f"COPY {table} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GeoJSON');"
        )