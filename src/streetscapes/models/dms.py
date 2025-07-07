# This code uses a pretrained model presented in the following research paper:
# Upchurch, Paul, and Ransen Niu. "A Dense Material Segmentation Dataset
# for Indoor and Outdoor Scene Parsing." ECCV 2022.
#
# Original repository: https://github.com/apple/ml-dms-dataset
#
# Copyright notice below covers the original research and model weights:
#
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import requests as rq
import torch
import numpy as np
import torchvision.transforms as tvt
import zipfile
import re

from streetscapes import utils
from streetscapes.models.base import PathLike
from streetscapes.models.base import ModelBase
from streetscapes import logger


class DMS(ModelBase):

    taxonomy = {
        2: "brickwork",
        9: "concrete",
        11: "engineered stone",
        12: "fabric",
        14: "foliage",
        18: "glass",
        19: "hair",
        22: "metal",
        24: "paint",
        25: "clear plastic",
        28: "opaque plastic",
        29: "rubber",
        30: "sand",
        31: "skin",
        32: "sky",
        34: "soil",
        35: "stone",
        37: "tile",
        39: "water",
        44: "wood",
        45: "asphalt",
    }

    inverse_taxonomy = {re.sub(r"\s+", " ", v): k for k, v in taxonomy.items()}

    def __init__(self, *args, **kwargs):

        # Initialise the base
        super().__init__(*args, **kwargs)

        # TODO: This needs to be converted to a proper HuggingFace-compatible model
        self.pt_archive_url = "https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms46_v1.zip"
        self.model_dir = utils.create_asset_dir("models", "DMS")
        self.pt_file = self.model_dir / "DMS46_v1.pt"
        self.model = None

        self._load_pretrained()

    def _load_pretrained(self, *args, **kwargs):

        if self.model is not None:
            return

        if not self.pt_file.exists():
            logger.info(f"Downloading model weights...")

            res = rq.get(self.pt_archive_url)

            pt_arch_file = self.pt_file.with_suffix(".zip")
            with open(pt_arch_file, "wb") as zf:
                zf.write(res.content)

            pt_arch = zipfile.ZipFile(pt_arch_file)
            pt_arch.extractall(path=self.pt_file.parent)
            pt_arch_file.unlink(missing_ok=True)
            logger.info(f"Done.")

        self.model = torch.jit.load(self.pt_file)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
