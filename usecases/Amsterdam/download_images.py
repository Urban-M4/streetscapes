#!/usr/bin/env python
# coding: utf-8

import sys

sys.path=['/gpfs/home4/cdonnely/Urban-M4/streetscapes/.venv/bin','','/home/cdonnely/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python312.zip','/home/cdonnely/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python3.12','/home/cdonnely/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/lib/python3.12/lib-dynload','/gpfs/home4/cdonnely/Urban-M4/streetscapes/.venv/lib/python3.12/site-packages','/gpfs/home4/cdonnely/Urban-M4/streetscapes/src']

from pathlib import Path
from environs import Env
import pandas as pd
import geopandas as gpd
import contextily
import glob

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

env = Env()
env.read_env(".env")
city = "Amsterdam"

hf_path = Path(env.path("HF_HOME"), city)
ws = SVWorkspace(hf_path)
mp = Mapillary(ws.env, root_dir=hf_path)

image_ids = glob.glob("./image_ids/*.json")
print(image_ids)
for f in image_ids:
    df = pd.read_json(f)
    ids = df["id"].to_list()
    urls = df["thumb_1024_url"].to_list()
    paths = mp.download_images(ids, urls)

#pdf = df.to_pandas()
#gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf.lon, pdf.lat))
#gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)


#image_ids = gdf.id
#urls = gdf.thumb_1024_url
#paths = mp.download_images(image_ids, urls)