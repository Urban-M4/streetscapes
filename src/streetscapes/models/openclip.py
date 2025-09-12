import numpy as np
from typing import Any
from PIL import Image

import open_clip

class OpenCLIP:
class OpenCLIPWrapper:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def classify(self, image: np.ndarray, labels: list[str]):
        pil_img = Image.fromarray(image)
        image_tensor = self.preprocess(pil_img).unsqueeze(0)
        text = self.tokenizer(labels)
        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        idx = int(torch.argmax(probs).item())
        return labels[idx]