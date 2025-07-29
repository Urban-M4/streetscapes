#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import glob

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace


city = "Amsterdam"

ws = SVWorkspace(city)
mp = Mapillary()

# TODO: get metadata from workspace
image_ids = glob.glob(f"{hf_path}/metadata/*.json")

print(image_ids)
for f in image_ids:
    df = pd.read_json(f)
    if "id" in df.columns:
        ids = df["id"].to_list()
        urls = df["thumb_1024_url"].to_list()
        paths = mp.download_images(ids, urls)
