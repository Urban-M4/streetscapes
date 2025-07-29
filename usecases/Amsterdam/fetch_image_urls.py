#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from environs import Env

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

env = Env()
env.read_env(".env")
city = "Amsterdam"

hf_path = Path(env.path("HF_HOME"), city)
ws = SVWorkspace(hf_path)
mp = Mapillary(ws.env, root_dir=hf_path)

# Fetch metadata by creator username
# df = mp.fetch_metadata_creator(creator_username="amsterdam", limit=1000)

# Fetch metadata by bounding box
bbox = [4.7,52.25,5.1,52.5]
records = mp.fetch_metadata_bbox(bbox, tile_size=0.1, limit=100, bbox_name="Amsterdam_")