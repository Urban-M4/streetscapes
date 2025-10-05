# streetscapes/sources/mapillary.py
from itertools import product
import math
from time import sleep
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point

Bbox = tuple[float, float, float, float]
"""(west, south, east, north)"""


class MapillaryClient:
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

    def fetch_metadata_tile(self, tile: Bbox, limit=1000):
        """Fetch metadata for one bounding box tile."""
        params = {
            "bbox": ",".join(map(str, tile)),
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": limit,
        }
        attempt = 0
        while attempt < self.retries:
            try:
                res = self.session.get(self.BASE_URL, params=params, timeout=10)
                res.raise_for_status()
                return res.json().get("data", [])
            except (requests.RequestException, ValueError):
                attempt += 1
                sleep(0.5 * attempt)
        return []

    def iter_tiles(
        self, bbox: Bbox, tile_size: float = 0.01
    ) -> Iterable[tuple[Bbox, str]]:
        """Yield (tile_bbox, tile_id) for a bounding box using numpy for edges."""
        west, south, east, north = bbox
        precision = max(0, -int(np.floor(np.log10(tile_size))) + 1)

        # Create longitude and latitude edges
        lon_starts = np.arange(west, east, tile_size)[:-1]
        lat_starts = np.arange(south, north, tile_size)[:-1]

        for w, s in product(lon_starts, lat_starts):
            e = w + tile_size
            n = s + tile_size
            tile = [
                round(float(w), precision),
                round(float(s), precision),
                round(float(e), precision),
                round(float(n), precision),
            ]
            tile_id = "_".join(f"{v:.{precision}f}" for v in tile)
            yield tile, tile_id

    def iter_metadata(self, bbox: Bbox, tile_size=0.01, limit=1000):
        """Yield (tile_id, DataFrame) for each tile."""

        for tile, tile_id in self.iter_tiles(bbox, tile_size):
            records = self.fetch_metadata_tile(tile, limit)

            if not records:
                continue

            df = _process_tile(records)
            yield tile_id, df

    def fetch_metadata(self, bbox: Bbox, tile_size=0.01, limit=1000):
        """Iterate over tiles and combine into a single dataframe."""
        df = pd.concat([df for _, df in self.iter_metadata(bbox, tile_size, limit)])

        # Convert to geopandas; use geometry as geometry
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
        return gdf.set_crs("EPSG:4326")


## Helpers
def _decimals_for_tile_size(tile_size: float) -> int:
    return max(0, -int(math.floor(math.log10(tile_size))) + 1)


def _unpack_geometry(geometry):
    """Extract geometry from mapillary metadata dict."""
    if isinstance(geometry, dict) and "coordinates" in geometry:
        # Using WKT makes it easy to ingest in geopandas and in duckdb later
        return Point(geometry["coordinates"]).wkt

    return None


def _process_tile(records):
    df = pd.DataFrame(records)

    df["geometry"] = df["geometry"].apply(_unpack_geometry)
    df["computed_geometry"] = df["computed_geometry"].apply(_unpack_geometry)

    return df


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from streetscapes.sources.mapillary import MapillaryClient

    load_dotenv()
    token = os.getenv("MAPILLARY_TOKEN")
    m = MapillaryClient(token)
    gdf = m.fetch_metadata(bbox=[4.89, 52.37, 4.91, 52.38])

    # Note: this is more realistic (but much more requests / data):
    # gdf = m.fetch_metadata(bbox=[4.89, 52.37, 4.91, 52.38], tile_size=0.001, limit=1000)
