"""Geospatial utils (bounding boxes and geo-hashing)."""

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # Delay slow imports for CLI responsiveness
    import shapely

Bbox = tuple[float, float, float, float]
"""west, south, easth, north"""


def split_bbox(
    bbox: Bbox, tile_size: float = 0.001
) -> tuple[int, Iterable[tuple[Bbox, str]]]:
    """Split bounding box into set of smaller tiles with fixed tile size."""
    import numpy as np

    west, south, east, north = bbox

    # Decimal precision for formatting
    precision = max(0, -int(np.floor(np.log10(tile_size))) + 1)

    # Convert coordinates to integer grid indices
    west_i = int(np.floor(west / tile_size))
    south_i = int(np.floor(south / tile_size))
    east_i = int(np.ceil(east / tile_size))
    north_i = int(np.ceil(north / tile_size))

    lon_indices = range(west_i, east_i)
    lat_indices = range(south_i, north_i)

    total = len(lon_indices) * len(lat_indices)

    def iter_tiles():
        for wi, si in product(lon_indices, lat_indices):
            w = wi * tile_size
            s = si * tile_size
            e = (wi + 1) * tile_size
            n = (si + 1) * tile_size

            tile = [
                round(w, precision),
                round(s, precision),
                round(e, precision),
                round(n, precision),
            ]

            tile_id = "_".join(f"{v:.{precision}f}" for v in tile)
            yield tile, tile_id

    return total, iter_tiles()


def get_geohash_shard_path(location: "shapely.Point"):
    """Get nested geo-hash path for given location given as a WKB point.

    Geo-hash precision from
    https://python-bloggers.com/2024/02/geohashing-from-scratch-in-python/
    Precision          Dimension
            1: 5,000km x 5,000km
            2:   1,250km x 625km
            3:     156km x 156km
            4:   31.9km x 19.5km
            5:   4.89km x 4.89km
            6:   1.22km x 0.61km
            7:       153m x 153m
            8:     38.2m x 19.1m
            9:     4.77m x 4.77m
           10:    1.19m x 0.596m
           11:     149mm x 149mm
           12:   37.2mm x 18.6mm
        Each level of precision subdivides the previous level into 32 subtiles.
    Shard path of precision 7, split in three parts abc/de/fg
    abc/ --> region level
    de/ --> neighbourhood scale (max 32x32 = 1024 per region)
    fg/ --> block level  (max 32x32 = 1024 per neighbourhood)
    """
    import pygeohash
    import shapely

    geom = shapely.from_wkb(location)  # type: ignore[call-overload]
    geohash = pygeohash.encode(geom.y, geom.x, precision=7)  # 153m x 153m
    return Path(geohash[:2]) / geohash[2:4] / geohash[4:6]
