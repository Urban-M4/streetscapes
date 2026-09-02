"""Segment Anything Model version 3."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tqdm import tqdm

from streetscapes import utils

if TYPE_CHECKING:
    import uuid


class SAM3:
    def __init__(
        self,
        weights: str | Path = "sam3.pt",
        device: str | None = None,
        confidence: float = 0.25,
        quantisation: str | None = None,
    ):
        """A SAM3 backend for StreetScapes.

        Uses the Ultralytics engine to allow for querying for multiple types of objects.

        IMPORTANT: The model weights need to be downloaded *manually*.
        To do that, you need to sign up to the SAM3 HuggingFace repository here:
        https://huggingface.co/facebook/sam3

        Args:
            weights: Path to the SAM3 model weights. Defaults to 'sam3.pt' in the
                current directory. This can be a symlink to the actual weights
                located elsewhere.
            device: Specify a device to run the model on.
            confidence: Confidence threshold for accepting segmentations.
            quantisation: Quantisation level. Possible values are `FP16` (faster
                inference) or `FP32`. `None` means that the default value will be used.
        """
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.device = utils.get_device(device)

        # Model parameters
        # ==================================================
        self.confidence = confidence
        self.quantisation = quantisation

        # SAM3 model
        # ==================================================
        # Overrides

        weight_path = Path(weights)

        if not weight_path.is_absolute():
            weight_path = Path(__file__).parent.resolve() / weight_path

        overrides = {
            "model": str(weight_path),
            "task": "segment",
            "mode": "predict",
            "save": False,
            "conf": confidence,
        }

        if isinstance(quantisation, str):
            overrides["quantize"] = quantisation.lower()

        self.model = SAM3SemanticPredictor(overrides=overrides)

    def segment_images(
        self,
        uids: list[uuid.UUID],
        images: list[np.ndarray],
        prompt: str | list[str],
    ) -> list[dict]:
        """Segment the provided sequence of images.

        Args:
            uids: UUID values associated with the images.
                This is used for keeping track of which images have been segmented,
                regardless of the file name and where they are stored.
            images: A list (batch) of images as NumPy arrays.
            prompt: A prompt as a string or a list of strings,
                indicating the categories (labels).

        Returns:
            A list of dictionaries containing instance-level segmentation information.
        """
        # Flatten the label dictionary
        _prompt = utils.extract_categories(prompt, as_list=True)

        segmentations = []

        for _, (uid, image) in tqdm(enumerate(zip(uids, images)), total=len(images)):
            # Dictionary that will hold all the information about the segmentation
            segmentation = {"uid": uid}

            self.model.set_image(Image.fromarray(image.astype(np.uint8)).convert("RGB"))

            # Segment the objects with SAM3
            # ==================================================
            result = self.model(text=_prompt)[0]

            # Process the model outputs
            # `result.masks` is `None` when no instances match the prompt.
            if result.masks is None:
                instance_labels = []
                instances = np.zeros((0, *image.shape[:2]), dtype=np.bool_)
            else:
                masks = result.masks.data.cpu().numpy()

                # Instance labels extracted from the class IDs
                instance_labels = [result.names[int(c)] for c in result.boxes.cls]

                # Populate the instance masks.
                instances = np.zeros(
                    (len(instance_labels), *image.shape[:2]),
                    dtype=np.bool_,
                )
                for mask_idx, sam_mask in enumerate(masks):
                    instances[mask_idx][sam_mask > 0] = True

            # Extract and store the segmentations.
            segmentation["labels"] = instance_labels
            segmentation["instances"] = instances
            segmentations.append(segmentation)

        return segmentations
