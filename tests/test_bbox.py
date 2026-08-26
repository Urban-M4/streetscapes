from streetscapes.utils.bbox import split_bbox


def test_split_bbox_basic():
    bbox = (0.0, 0.0, 0.01, 0.01)
    total, tiles = split_bbox(bbox, tile_size=0.001)

    tiles = list(tiles)
    assert total == len(tiles)
    assert total == 100 # 10x10 grid
    assert all(len(t[0]) == 4 for t in tiles)
    assert all(isinstance(t[1], str) for t in tiles)
