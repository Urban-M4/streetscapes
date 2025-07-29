#!/usr/bin/env python
# coding: utf-8

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

mp = Mapillary()

# Fetch metadata by creator username
# df = mp.fetch_metadata_creator(creator_username="amsterdam", limit=100)

# Fetch metadata by bounding box
bbox = [4.7, 52.25, 5.1, 52.5]
records = mp.fetch_metadata_bbox(bbox, tile_size=0.1, limit=10)

# Store metadata in a workspace
ws = SVWorkspace("Amsterdam")
ws.save_metadata(records)
