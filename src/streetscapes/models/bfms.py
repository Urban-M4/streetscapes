from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)


class BFMS:
    """Building/Facade Material Segmentation model
    based on Mask2Former.
    """

    def __init__(self, model_path: Path = Path("./trained_model"), device: str = None):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + processor
        self._load_model()

    def _load_model(self):
        config = Mask2FormerConfig.from_pretrained(self.model_path / "config.json")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_path / "model.safetensors", config=config
        ).to(self.device)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path / "config.json", use_fast=True
        )

    def segment(self, image: np.ndarray | Image.Image) -> dict[str, Any]:
        """Run BFMS segmentation.

        Args:
            image: Input image as numpy array or PIL Image.

        Returns:
            dict with:
                - mask: np.ndarray [H, W], semantic labels
                - labels: list[str], predicted class names

        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits
        mask_logits = outputs.masks_queries_logits[0]  # [Q, H, W]
        class_logits = outputs.class_queries_logits[0]  # [Q, C]

        # Convert to probabilities
        masks_probs = torch.sigmoid(mask_logits)
        class_probs = torch.softmax(class_logits, dim=-1)

        # Combine: [C, H, W]
        pixel_class_probs = torch.einsum("qc,qhw->chw", class_probs, masks_probs)

        # Argmax → semantic mask
        semantic_mask = torch.argmax(pixel_class_probs, dim=0).cpu().numpy()

        # Convert to dict
        return {
            "mask": semantic_mask,
            "labels": [f"class_{i}" for i in np.unique(semantic_mask)],
        }


label_colors = np.array(
    [
        (0, 0, 0),
        (139, 69, 19),
        (205, 133, 63),
        (178, 34, 34),
        (210, 180, 140),
        (34, 139, 34),
        (255, 165, 0),
        (255, 215, 0),
        (0, 0, 128),
        (128, 128, 128),
        (192, 192, 192),
        (255, 105, 180),
        (139, 0, 0),
        (75, 0, 130),
        (0, 191, 255),
        (70, 130, 180),
        (0, 128, 0),
        (255, 20, 147),
        (160, 82, 45),
        (184, 134, 11),
        (0, 255, 255),
        (255, 192, 203),
        (0, 100, 0),
        (176, 224, 230),
        (139, 69, 19),
        (205, 92, 92),
        (192, 192, 192),
        (255, 250, 250),
        (255, 0, 255),
        (173, 216, 230),
        (255, 228, 196),
        (245, 245, 245),
        (255, 239, 213),
        (135, 206, 250),
        (105, 105, 105),
        (128, 0, 128),
        (194, 178, 128),
        (255, 182, 193),
        (135, 206, 235),
        (255, 250, 250),
        (128, 128, 0),
        (139, 69, 19),
        (169, 169, 169),
    ],
    dtype=np.uint8,
)

# Label map
id2label = {
    0: "Background/Unclassified",
    1: "Wood/Bamboo",
    2: "Ground tile",
    3: "Brick",
    4: "Cardboard/Paper",
    5: "Tree",
    6: "Roof tile",
    7: "Ceramic",
    8: "Chalkboard/Blackboard",
    9: "Asphalt",
    10: "Cement/Concrete",
    11: "Composite decorative board",
    12: "Rammed earth",
    13: "Fabric/Cloth",
    14: "Water",
    15: "Windows with metal fences",
    16: "Foliage",
    17: "Food",
    18: "Fur",
    19: "Pottery",
    20: "Glass",
    21: "Hair",
    22: "Roofing waterproof material",
    23: "Ice",
    24: "Leather",
    25: "Carved brick",
    26: "Metal",
    27: "Mirror",
    28: "Enamel",
    29: "Paint/Coating/Plaster",
    30: "Window screen",
    31: "Whiteboard",
    32: "Photograph/Painting/Airbrushed fabric",
    33: "Plastic, clear",
    34: "Plastic, non-clear",
    35: "Rubber/Latex",
    36: "Sand",
    37: "Skin/Lips",
    38: "Sky",
    39: "Snow",
    40: "Engineered Stone/Imitation Stone",
    41: "Soil/Mud",
    42: "Natural Stone",
}
