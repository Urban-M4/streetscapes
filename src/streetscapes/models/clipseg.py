import numpy as np
from typing import Any
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
import torch
from PIL import Image
import torchvision as tv


class CLIPSeg:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompt = [prompt]
        pil_img = Image.fromarray(image)
        inputs = self.processor(
            text=prompt,
            images=[pil_img] * len(prompt),
            return_tensors="pt",
        )
        logits = self.model(**inputs).logits
        mask = torch.sigmoid(logits)
        rs = tv.transforms.Resize(image.shape[:2])
        mask = rs(mask).cpu().numpy()
        filler = np.zeros(mask.shape[:3])
        return np.stack([mask, filler, filler], axis=-1)
