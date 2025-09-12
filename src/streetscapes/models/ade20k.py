# streetscapes/models/ade20k.py

import torch
import torch.nn as nn
import torchvision
from mit_semseg.models import ModelBuilder, SegmentationModule
import numpy as np


class ADE20KFacade:
    """Atomic ADE20K segmentation model.

    From ZFMS pipeline: https://github.com/Nadatarkhan/Zero-shot-Facade-Material-Segmentation
    """

    def __init__(
        self, encoder_weights: str, decoder_weights: str, device: torch.device
    ):
        net_encoder = ModelBuilder.build_encoder(
            arch="resnet50dilated", fc_dim=2048, weights=encoder_weights
        )
        net_decoder = ModelBuilder.build_decoder(
            arch="ppm_deepsup",
            fc_dim=2048,
            num_class=150,
            weights=decoder_weights,
            use_softmax=True,
        )
        crit = nn.NLLLoss(ignore_index=-1)
        self.seg_module = (
            SegmentationModule(net_encoder, net_decoder, crit).eval().to(device)
        )
        self.device = device

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def segment(self, image_np: np.ndarray, building_min_fraction: float = 0.2):
        """Segment image and return ADE-masked image and empty mask."""
        h, w, _ = image_np.shape
        img_tensor = self.transform(image_np).unsqueeze(0).to(self.device)

        scores = self.seg_module({"img_data": img_tensor}, segSize=(h, w))
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu().numpy()[0]

        # ADE20K building mask (1=building, 25=wall)
        ade_mask = np.logical_or(pred == 1, pred == 25).astype(np.uint8)
        building_fraction = ade_mask.sum() / (h * w)
        if building_fraction < building_min_fraction:
            raise ValueError("Not enough building in the scene.")

        empty_mask = (1 - ade_mask).astype(bool)
        masked_rgb = (image_np * np.repeat(ade_mask[:, :, None], 3, axis=2)).astype(
            np.uint8
        )

        # Return manifest as dictionary
        manifest = {"ade_mask": ade_mask}

        return masked_rgb, empty_mask, manifest
