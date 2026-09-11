import pytest

from streetscapes.utils.geo import split_bbox


def test_split_bbox_basic():
    bbox = (0.0, 0.0, 0.01, 0.01)
    total, tiles = split_bbox(bbox, tile_size=0.001)

    tiles = list(tiles)
    assert total == len(tiles)
    assert total == 100 # 10x10 grid
    assert all(len(t[0]) == 4 for t in tiles)
    assert all(isinstance(t[1], str) for t in tiles)


def test_split_bbox_clips_to_the_bbox():
    """A box smaller than a tile must not be widened to the whole grid cell.

    An unclipped tile would ask the source for a far larger area than was
    requested, and spend any per-tile limit on images outside the box.
    """
    bbox = (2.336698, 48.865696, 2.346236, 48.870650)
    total, tiles = split_bbox(bbox, tile_size=0.05)

    tiles = list(tiles)
    assert total == 1
    assert tiles[0][0] == [*bbox]
    # The ID still names the grid cell, so it is stable across runs.
    assert tiles[0][1] == "2.300_48.850_2.350_48.900"


def test_split_bbox_covers_the_bbox_exactly():
    """Tiles must together cover the bounding box, without spilling outside it."""
    west, south, east, north = bbox = (2.20, 48.80, 2.45, 48.92)
    total, tiles = split_bbox(bbox, tile_size=0.05)

    tiles = [t for t, _ in tiles]
    assert total == len(tiles)

    covered = sum((t[2] - t[0]) * (t[3] - t[1]) for t in tiles)
    assert covered == pytest.approx((east - west) * (north - south))
    assert all(
        t[0] >= west and t[1] >= south and t[2] <= east and t[3] <= north for t in tiles
    )
