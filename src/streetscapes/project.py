# streetscapes/project.py
from pathlib import Path
import ibis

class Project:
    """
    A project manages persistence of imagery and analysis metadata
    in a DuckDB database. It is source-agnostic: it does not know
    about Mapillary, Kartaview, or other clients. It only provides
    ingestion, table creation, and recovery tracking.
    """

    def __init__(self, path: Path, create: bool = True):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "streetscapes.duckdb"
        self.con = ibis.duckdb.connect(str(self.db_path))
        if create:
            self._init_db()

    # ------------------------------------------------------------
    # Core initialization
    # ------------------------------------------------------------
    def _init_db(self):
        """Load spatial extension and create core tables."""
        self.con.raw_sql("INSTALL spatial; LOAD spatial;")
        # Images table: explicit schema
        self.ensure_table_from_schema(
            "images",
            """
            id TEXT PRIMARY KEY,
            source TEXT,
            path TEXT,
            geometry GEOMETRY,
            captured_at TIMESTAMP
            """,
        )

    # ------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------
    def ensure_table_from_schema(self, table_name: str, schema: str):
        """Ensure a table exists with an explicit SQL schema."""
        try:
            self.con.table(table_name)
        except Exception:
            self.con.raw_sql(f"CREATE TABLE {table_name} ({schema})")

    def ensure_table_from_dataframe(self, table_name: str, df):
        """Ensure a table exists, creating it from the DataFrame schema."""
        try:
            self.con.table(table_name)
        except Exception:
            self.con.create_table(df, table_name=table_name)

    # ------------------------------------------------------------
    # Processed tile tracking (for resumable ingestion)
    # ------------------------------------------------------------
    def get_processed_tiles(self, source: str) -> set[str]:
        """Return set of processed tile IDs for this source."""
        self.ensure_table_from_schema(
            f"{source}_processed_tiles", "tile_id TEXT PRIMARY KEY"
        )
        rows = self.con.raw_sql(
            f"SELECT tile_id FROM {source}_processed_tiles"
        ).fetchall()
        return {r[0] for r in rows}

    def mark_tile_processed(self, source: str, tile_id: str):
        """Mark a tile as processed (idempotent)."""
        self.ensure_table_from_schema(
            f"{source}_processed_tiles", "tile_id TEXT PRIMARY KEY"
        )
        self.con.raw_sql(
            f"INSERT OR REPLACE INTO {source}_processed_tiles VALUES (?)", (tile_id,)
        )

    # ------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------
    def ingest_metadata_batch(self, source: str, df, tile_id: str, table: str):
        """
        Insert a metadata batch (DataFrame) into the given table,
        and record that the tile_id has been processed.
        """
        if df.empty:
            return
        # Create table if missing (dynamic schema)
        self.ensure_table_from_dataframe(table, df)
        # Insert batch
        self.con.create_table("batch_view", df, overwrite=True)
        self.con.raw_sql(f"INSERT INTO {table} SELECT * FROM batch_view")
        # Track tile
        self.mark_tile_processed(source, tile_id)
