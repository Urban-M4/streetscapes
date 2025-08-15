#!/usr/bin/env python
# coding: utf-8

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

ws = SVWorkspace("amsterdam")
metadata = ws.load_metadata("subset_lcz_kittner_081.parquet")

mp = Mapillary()
ids = metadata["id"].to_list()
urls = metadata["thumb_2048_url"].to_list()
paths = mp.download_images(ids, urls)