#!/usr/bin/env python
# coding: utf-8

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

mp = Mapillary()

# Fetch metadata by creator username
# df = mp.fetch_image_ids_creator(creator_username="amsterdam", limit=1000)

# Fetch metadata by bounding box
bbox = [4.7,52.25,5.1,52.5]
records = mp.fetch_image_ids_bbox(bbox, tile_size=0.1, limit=10, bbox_name="Amsterdam_")

# TODO: Collect all metadata into a single parquet/csv/duckcb file inside the workspace
ws = SVWorkspace("Amsterdam")
