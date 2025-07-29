#!/usr/bin/env python
# coding: utf-8

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

# Use a dedicated workpace for this use case
ws = SVWorkspace("Amsterdam")
mp = Mapillary(root_dir=ws.root_dir)  # TODO: remove env argument from (Image)SourceBase

# Fetch metadata by creator username
# df = mp.fetch_image_ids_creator(creator_username="amsterdam", limit=1000)

# Fetch metadata by bounding box
bbox = [4.7,52.25,5.1,52.5]
records = mp.fetch_image_ids_bbox(bbox, tile_size=0.1, limit=100, bbox_name="Amsterdam_")

# TODO: Collect all metadata into a single parquet/csv/duckcb file inside the workspace
