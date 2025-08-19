#!/usr/bin/env python
# coding: utf-8

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

# Area of interest
# area = dict(name="wageningen", bbox=[5.63, 51.95, 5.69, 51.98])
area = dict(name="amsterdam", bbox=[4.7, 52.25, 5.1, 52.5])
# area = dict(name="enschede", bbox=[6.81, 52.17, 6.94, 52.24])

# Fetch image ids in area of interest
mp = Mapillary()
metadata = mp.fetch_metadata_bbox(bbox=area["bbox"], tile_size=0.01, limit=500)

# Store metadata in a workspace
ws = SVWorkspace(area["name"])
ws.save_metadata(metadata, filename="mapillary.parquet")