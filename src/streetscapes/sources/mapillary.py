from pathlib import Path
from typing import List, Protocol, Any
from time import sleep
import math
import requests
from rich.progress import track
import json
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq


class OutputWriter(Protocol):
    """
    Protocol for writing batches of metadata to storage.

    Methods:
        init(path: Path) -> None:
            Initialize storage (e.g., create empty file if missing).
        append(records: List[dict], path: Path) -> None:
            Append a batch of records to storage.
        read(path: Path) -> Any:
            Read the entire stored dataset.
    """

    def init(self, path: Path) -> None: ...

    def append(self, records: List[dict], path: Path) -> None: ...

    def read(self, path: Path) -> Any: ...


class Mapillary:
    """
    Mapillary street view interface with incremental, crash-safe metadata fetching.

    Attributes:
        token: Mapillary OAuth token.
        retries: number of retries per tile in case of request failure.
        session: authenticated requests session.
    """

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
        """
        Initialize MapillarySource.

        Args:
            token: Mapillary OAuth token.
            retries: number of retries for failed API requests.
        """
        self.token = token
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {self.token}"})

    def get_image_url(
        self, image_id: str, resolution: str = "thumb_2048_url"
    ) -> str | None:
        """
        Retrieve the image URL for a given image ID.

        Args:
            image_id: Mapillary image ID.
            resolution: field specifying image resolution, default is "thumb_2048_url".

        Returns:
            URL string or None if request fails.
        """
        url = f"{self.BASE_URL}/{image_id}?fields={resolution}"
        try:
            res = self.session.get(url, timeout=10)
            res.raise_for_status()
            return res.json().get(resolution)
        except requests.RequestException:
            return None

    @staticmethod
    def _decimals_for_tile_size(tile_size: float) -> int:
        """
        Determine decimal rounding precision for a given tile size in degrees.
        Ensures consistent raster snapping and avoids duplicate tiles.

        Examples:
            tile_size=0.01  -> precision=3
            tile_size=0.001 -> precision=4
        """
        return max(0, -int(math.floor(math.log10(tile_size))) + 1)

    @classmethod
    def split_bbox(
        cls, bbox: List[float], tile_size: float = 0.01
    ) -> List[List[float]]:
        """
        Split a bounding box into smaller tiles.

        Args:
            bbox: [west, south, east, north]
            tile_size: size of each tile in degrees

        Returns:
            List of [west, south, east, north] tiles.
        """
        west, south, east, north = bbox
        precision = cls._decimals_for_tile_size(tile_size)
        tiles = []
        lon_steps = int((east - west) / tile_size + 1)
        lat_steps = int((north - south) / tile_size + 1)
        for i in range(lon_steps):
            for j in range(lat_steps):
                w = round(west + i * tile_size, precision)
                s = round(south + j * tile_size, precision)
                e = round(min(w + tile_size, east), precision)
                n = round(min(s + tile_size, north), precision)
                tiles.append([w, s, e, n])
        return tiles

    def fetch_metadata_tile(self, tile: List[float]) -> List[dict]:
        """
        Fetch metadata for a single tile.

        Args:
            tile: bounding box [west, south, east, north] for this tile.

        Returns:
            List of metadata records (JSON/dict) for this tile.
        """
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
            except (requests.RequestException, json.JSONDecodeError):
                attempt += 1
                sleep(0.5 * attempt)
        print(f"Tile {tile} failed after {self.retries} attempts.")
        return []

    def fetch_metadata(
        self,
        bbox: List[float],
        tile_size: float,
        output_file: Path,
        writer: OutputWriter | None = None,
    ) -> Any:
        """
        Fetch Mapillary metadata for a bounding box and write incrementally.

        Args:
            bbox: bounding box [west, south, east, north]
            tile_size: size of each tile in degrees
            output_file: path to output file
            writer: OutputWriter instance (default: PyArrowWriter)

        Returns:
            Final combined dataset, depending on writer backend.
        """
        writer = writer or PyArrowGeoParquetWriter()
        writer.init(output_file)

        state_file = output_file.with_suffix(".state.json")
        if state_file.exists():
            with open(state_file) as f:
                processed_tiles = set(json.load(f))
        else:
            processed_tiles = set()

        tiles = self.split_bbox(bbox, tile_size)
        precision = self._decimals_for_tile_size(tile_size)

        for tile in track(tiles, description="Fetching Mapillary tiles..."):
            tile_id = "_".join(f"{v:.{precision}f}" for v in tile)
            if tile_id in processed_tiles:
                continue

            records = self.fetch_metadata_tile(tile)
            writer.append(records, output_file)

            processed_tiles.add(tile_id)
            with open(state_file, "w") as f:
                json.dump(list(processed_tiles), f)

        return writer.read(output_file)


GEO_METADATA = {
    "primary_column": "geometry",
    "columns": {
        "geometry": {"encoding": "WKB", "geometry_type": "Point", "crs": "EPSG:4326"}
    },
}


class PyArrowGeoParquetWriter:
    """
    Efficient writer for GeoParquet using PyArrow.

    Converts Mapillary JSON geometries to WKB and writes
    Arrow tables directly to Parquet with GeoParquet metadata.
    Supports incremental appends.
    """

    def init(self, path: Path) -> None:
        if not path.exists():
            empty_table = pa.table(
                {}, metadata={b"geo": json.dumps(GEO_METADATA).encode("utf-8")}
            )
            pq.write_table(empty_table, path)

    def append(self, records: List[dict], path: Path) -> None:
        if not records:
            return

        # Convert geometry dict → WKB
        for rec in records:
            geom = rec.get("geometry")
            if geom and "coordinates" in geom:
                rec["geometry"] = Point(geom["coordinates"]).wkb
            else:
                rec["geometry"] = None

        batch = pa.Table.from_pylist(records)
        batch = batch.replace_schema_metadata(
            {b"geo": json.dumps(GEO_METADATA).encode("utf-8")}
        )

        # Read existing table if file exists and is non-empty
        if path.exists() and path.stat().st_size > 0:
            try:
                existing = pq.read_table(path)
                combined = pa.concat_tables([existing, batch])
            except Exception:
                combined = batch
        else:
            combined = batch

        pq.write_table(combined, path)

    def read(self, path: Path):
        return pq.read_table(path)


class GeoPandasGeoParquetWriter:
    """
    Writer for GeoParquet using GeoPandas.

    Converts JSON geometries to Shapely Points and writes
    GeoDataFrames to Parquet with CRS EPSG:4326.
    Supports incremental appends.
    """

    def init(self, path: Path) -> None:
        if not path.exists():
            gpd.GeoDataFrame().to_parquet(path)

    def append(self, records: List[dict], path: Path) -> None:
        if not records:
            return

        df = pd.DataFrame(records)
        df["geometry"] = df["geometry"].apply(
            lambda c: Point(c["coordinates"]) if c else None
        )
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        if path.exists():
            gdf_existing = gpd.read_parquet(path)
            gdf = gpd.GeoDataFrame(
                pd.concat([gdf_existing, gdf], ignore_index=True),
                geometry="geometry",
                crs="EPSG:4326",
            )

        gdf.to_parquet(path)

    def read(self, path: Path):
        return gpd.read_parquet(path)
