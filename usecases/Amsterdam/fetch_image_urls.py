#!/usr/bin/env python
# coding: utf-8

import sys

sys.path=['/gpfs/home4/cdonnely/Urban-M4/streetscapes/.venv/bin','','/home/cdonnely/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python312.zip','/home/cdonnely/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python3.12','/home/cdonnely/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python3.12/lib-dynload','/gpfs/home4/cdonnely/Urban-M4/streetscapes/.venv/lib/python3.12/site-packages','/gpfs/home4/cdonnely/Urban-M4/streetscapes/src']

from pathlib import Path
from environs import Env
import geopandas as gpd
import contextily

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

env = Env()
env.read_env(".env")
city = "Amsterdam"

hf_path = Path(env.path("HF_HOME"), city)
ws = SVWorkspace(hf_path)
mp = Mapillary(ws.env, root_dir=hf_path)

df = mp.fetch_image_ids_creator(creator_username="amsterdam", limit=1000)
