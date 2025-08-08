#!/usr/bin/env python
# coding: utf-8

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

ws = SVWorkspace("wageningen")
metadata = ws.load_metadata()

mp = Mapillary()
ids = metadata["id"].to_list()
urls = metadata["thumb_1024_url"].to_list()
paths = mp.download_images(ids, urls)