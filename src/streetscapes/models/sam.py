import numpy as np


class SAM:
    def __init__(self, model_id="facebook/sam2.1-hiera-large", device=None):
        self.device = device
        self.model = None  # load model here

    def segment(self, images: np.ndarray, boxes=None) -> list[np.ndarray]:
        # Dummy implementation
        return [np.zeros_like(images[..., 0], dtype=np.uint8)]
