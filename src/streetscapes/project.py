class Project:
    """Minimal project managing a DuckDB/Ibis connection."""

    def __init__(self, db_path: str = "streetscapes.duckdb"):
        import ibis

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
        import ibis
        from shapely.geometry import box

        table = self.get_table(table_name)
        envelope_expr = ibis.literal(box(*bbox).wkt, type="geospatial:geometry")
        return table.filter(table.geometry.within(envelope_expr))
