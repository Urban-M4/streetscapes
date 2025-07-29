#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from environs import Env
import pandas as pd
import glob

from streetscapes.sources import Mapillary
from streetscapes.streetview import SVWorkspace

env = Env()
env.read_env(".env")
city = "Amsterdam"

hf_path = Path(env.path("HF_HOME"), city)
ws = SVWorkspace(hf_path)
mp = Mapillary(ws.env, root_dir=hf_path)

image_ids = glob.glob(f"{hf_path}/metadata/*.json")
print(image_ids)
for f in image_ids:
    df = pd.read_json(f)
    if "id" in df.columns:
        ids = df["id"].to_list()
        urls = df["thumb_1024_url"].to_list()
        paths = mp.download_images(ids, urls)
