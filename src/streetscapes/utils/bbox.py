from itertools import product
from typing import Iterable

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
