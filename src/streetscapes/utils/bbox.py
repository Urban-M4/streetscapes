from itertools import product

import numpy as np

from typing import Iterable

Bbox = tuple[float, float, float, float]
"""west, south, easth, north"""


def split_bbox(
    bbox: Bbox, tile_size: float = 0.01
) -> tuple[int, Iterable[tuple[Bbox, str]]]:
    """Split bounding box into set of smaller tiles with fixed tile size."""
    west, south, east, north = bbox
    precision = max(0, -int(np.floor(np.log10(tile_size))) + 1)

    # Snap bbox to tile raster
    west_snapped = np.floor(west / tile_size) * tile_size
    south_snapped = np.floor(south / tile_size) * tile_size
    east_snapped = np.ceil(east / tile_size) * tile_size
    north_snapped = np.ceil(north / tile_size) * tile_size

    # Create longitude and latitude edges
    lon_starts = np.arange(west_snapped, east_snapped + tile_size, tile_size)[:-1]
    lat_starts = np.arange(south_snapped, north_snapped + tile_size, tile_size)[:-1]

    total = len(lon_starts) * len(lat_starts)

    def iter_tiles():
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

    return total, iter_tiles()
