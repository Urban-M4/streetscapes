import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image
from streetscapes import utils
from streetscapes.config import CFG


class BFMS:
    """Building/Facade Material Segmentation model based on Mask2Former."""

    def __init__(
        self,
        device: str | None = None,
        model_id: str = "jinfengxie/BFMS_1014",
    ):
        """Load the BFMS model.

        Args:
            device: Specify a device to run the model on.
        """
        import transformers as tform

        self.device = utils.get_device(device)

        config = tform.Mask2FormerConfig.from_pretrained(model_id)

        self.model = tform.Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id,
            config=config,
        ).to(self.device)

        # won't load directly from BFMS ID
        # from_pretrained should accept URL but this is broken in 
        # transformers v5
        tmp_model_dir = CFG.image_dir / "models"
        tmp_model_dir.mkdir(exist_ok=True)
        conf_path =  tmp_model_dir / "bfms-config.json"
        config.to_json_file(conf_path)

        self.processor = tform.AutoImageProcessor.from_pretrained(
            conf_path,
            use_fast=True,
        )

        self.model.eval()

    def segment(self, image: np.ndarray) -> dict[str, Any]:
        """Run BFMS segmentation.

        Args:
            image: Input image as numpy array.

        Returns:
            semantic_mask: np.ndarray [H, W], predicted class ids
        """
        # Convert to RGB
        rgb_image = Image.fromarray(image).convert("RGB")

        # Preprocess
        inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)

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

        # Split the segmentation into labels and instance masks.
        labels = set(np.unique(semantic_mask).tolist())
        labels.discard(0)  # Ignore background
        instances = np.zeros((len(labels), *semantic_mask.shape), dtype=np.bool_)
        for idx, lbl in enumerate(labels):
            instances[idx][semantic_mask == lbl] = True

        return {
            "labels": [id2label[l - 1] for l in labels],
            "instances": instances,
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


def md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
