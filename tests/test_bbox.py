from streetscapes.utils.bbox import split_bbox


def test_split_bbox_basic():
    bbox = (0.0, 0.0, 0.02, 0.02)
    total, tiles = split_bbox(bbox, tile_size=0.01)

    tiles = list(tiles)
    assert total == len(tiles)
    assert total == 4  # 2x2 grid
    assert all(len(t[0]) == 4 for t in tiles)
    assert all(isinstance(t[1], str) for t in tiles)
