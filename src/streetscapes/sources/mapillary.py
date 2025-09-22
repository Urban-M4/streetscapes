from streetscapes.utils.logging import logger
from pathlib import Path
from typing import List
from time import sleep
import math
import requests
import pandas as pd
import geopandas as gpd
import duckdb
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps
from rich.progress import track


class DuckDBManifest:
    """
    Incremental cache for Mapillary metadata in DuckDB using spatial extension.
    """

    def __init__(self, path: Path):
        self.path = path
        self.con = duckdb.connect(str(path))

        # Load spatial extension for GEOMETRY support
        self.con.execute("INSTALL spatial;")
        self.con.execute("LOAD spatial;")

        self.first_batch = True
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS processed_tiles (tile_id VARCHAR PRIMARY KEY)"
        )

    def get_processed_tiles(self) -> set:
        return set(
            row[0]
            for row in self.con.execute(
                "SELECT tile_id FROM processed_tiles"
            ).fetchall()
        )

    def add_batch(self, gdf: gpd.GeoDataFrame, tile_id: str):
        gdf = gdf.copy()
        # Geometry is already parsed as shapely Point in GeoDataFrame
        # Convert geometry to WKB BLOB (bytes)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: wkb_dumps(geom) if geom is not None else None
        )

        if self.first_batch:
            self.con.register("gdf_view", gdf)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS metadata AS
                SELECT * EXCLUDE geometry, ST_GeomFromWKB(geometry) AS geometry
                FROM (SELECT * FROM gdf_view)
            """)
            self.first_batch = False
        else:
            self.con.register("gdf_view", gdf)
            self.con.execute("""
                INSERT INTO metadata
                SELECT * EXCLUDE geometry, ST_GeomFromWKB(geometry) AS geometry
                FROM gdf_view
            """)

        self.con.execute("INSERT OR IGNORE INTO processed_tiles VALUES (?)", [tile_id])


class Mapillary:
    BASE_URL = "https://graph.mapillary.com/images"
    DEFAULT_FIELDS = [
        "id",
        "geometry",
        "captured_at",
        "sequence",
        "thumb_2048_url",
        "altitude",
        "compass_angle",
        "computed_altitude",
        "computed_geometry",
    ]

    def __init__(self, token: str, retries: int = 3):
        self.token = token
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {self.token}"})

    def fetch_metadata_tile(self, tile: List[float]) -> List[dict]:
        params = {
            "bbox": ",".join(map(str, tile)),
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": 1000,
        }
        attempt = 0
        while attempt < self.retries:
            try:
                res = self.session.get(self.BASE_URL, params=params, timeout=10)
                res.raise_for_status()
                return res.json().get("data", [])
            except (requests.RequestException, ValueError) as e:
                attempt += 1
                logger.warning(f"Tile {tile} request failed (attempt {attempt}): {e}")
                sleep(0.5 * attempt)
        logger.error(f"Tile {tile} failed after {self.retries} attempts.")
        return []

    @staticmethod
    def _decimals_for_tile_size(tile_size: float) -> int:
        return max(0, -int(math.floor(math.log10(tile_size))) + 1)

    def iter_tiles(self, bbox: List[float], tile_size: float):
        """
        Generator yielding (tile_bbox, tile_id) for a bounding box.
        """
        west, south, east, north = bbox
        precision = self._decimals_for_tile_size(tile_size)
        lon_steps = int((east - west) / tile_size + 1)
        lat_steps = int((north - south) / tile_size + 1)

        for i in range(lon_steps):
            for j in range(lat_steps):
                w = round(west + i * tile_size, precision)
                s = round(south + j * tile_size, precision)
                e = round(min(w + tile_size, east), precision)
                n = round(min(s + tile_size, north), precision)
                tile = [w, s, e, n]
                tile_id = "_".join(f"{v:.{precision}f}" for v in tile)
                yield tile, tile_id

    def fetch_metadata(
        self,
        bbox: List[float],
        tile_size: float,
        output_file: Path,
    ) -> gpd.GeoDataFrame:
        """
        Fetch Mapillary metadata incrementally, exporting to a duckDB manifest file.
        """
        logger.info(
            f"Preparing to fetch metadata for bbox={bbox}, tile_size={tile_size}, output_file={output_file}"
        )
        manifest = DuckDBManifest(output_file)
        processed_tiles = manifest.get_processed_tiles()

        for tile, tile_id in track(
            self.iter_tiles(bbox, tile_size), description="Fetching Mapillary tiles..."
        ):
            if tile_id in processed_tiles:
                logger.info(f"Tile {tile_id} already processed. Skipping.")
                continue

            records = self.fetch_metadata_tile(tile)
            if not records:
                logger.warning(f"No records for tile {tile_id}.")
                continue

            # Convert records to GeoDataFrame using computed_geometry if present, else geometry, else None
            df = pd.DataFrame(records)

            def pick_geometry(row):
                cg = row.get("computed_geometry")
                if isinstance(cg, dict) and "coordinates" in cg:
                    return Point(cg["coordinates"])
                g = row.get("geometry")
                if isinstance(g, dict) and "coordinates" in g:
                    return Point(g["coordinates"])
                return None

            df["geometry"] = df.apply(pick_geometry, axis=1)
            if df["geometry"].isnull().all():
                logger.warning(
                    f"All geometry values are null for tile {tile_id}. This may indicate an empty or invalid API response."
                )
                # TODO: sometimes I still get "geometry column does not contain geometry"
            gdf = gpd.GeoDataFrame(
                df.drop(columns=["computed_geometry"]),
                geometry="geometry",
                crs="EPSG:4326",
            )
            gdf["tile_id"] = tile_id

            manifest.add_batch(gdf, tile_id)
