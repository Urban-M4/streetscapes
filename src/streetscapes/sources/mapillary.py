# streetscapes/sources/mapillary.py
import math
import requests
from time import sleep
import pandas as pd
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

    def iter_tiles(self, bbox: Bbox, tile_size=0.01):
        """Yield (tile_bbox, tile_id) for a bounding box."""
        west, south, east, north = bbox
        precision = _decimals_for_tile_size(tile_size)
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

    def iter_metadata(self, bbox: Bbox, tile_size=0.01, limit=1000):
        """Yield (tile_id, DataFrame) for each tile."""

        for tile, tile_id in self.iter_tiles(bbox, tile_size):
            records = self.fetch_metadata_tile(tile, limit)

            if not records:
                continue

            df = _process_tile(records)
            yield tile_id, df

    def fetch_metadata(self, bbox: Bbox, tile_size=0.01, limit=1000):
        """Fetch all tiles and combine into a single dataframe."""
        return pd.concat([df for _, df in self.iter_metadata(bbox, tile_size, limit)])


## Helpers
def _decimals_for_tile_size(tile_size: float) -> int:
    return max(0, -int(math.floor(math.log10(tile_size))) + 1)


def _unpack_geometry(geometry):
    """Extract geometry from mapillary metadata dict."""
    if isinstance(geometry, dict) and "coordinates" in geometry:
        return Point(geometry["coordinates"])

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
    df = m.fetch_metadata(bbox=[4.89, 52.37, 4.91, 52.38])
