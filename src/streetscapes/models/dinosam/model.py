import numpy as np
import uuid
from PIL import Image
from tqdm import tqdm
from streetscapes import utils
from streetscapes.utils import logger


class DinoSAM:

    def __init__(
        self,
        sam_model_id: str = "facebook/sam2.1-hiera-large",
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        """A model combining SAM2 and GroundingDINO for promptable instance segmentation.
        Inspired by [LangSAM](https://github.com/luca-medeiros/lang-segment-anything) and [SamGeo](https://samgeo.gishub.org/samgeo/).

        Args:
            sam_model_id: SAM2 model. Possible options include:
                - facebook/sam2.1-hiera-tiny
                - facebook/sam2.1-hiera-small
                - facebook/sam2.1-hiera-large
            dino_model_id: A GroundingDINO model.
            box_threshold: This parameter is used for modulating the identification of objects in the images.
                The box threshold is related to the model confidence,
                so a higher value makes the model more selective because
                it is equivalent to requiring the model to only select
                objects that it feels confident about.
            text_threshold: This parameter is also used for influencing the selectivity of the model
                by requiring a stronger association between the prompt and the segment.
            device: Specify a device to run the model on.

        """
        import transformers

        self.device = utils.get_device(device)

        # Model parameters
        # ==================================================
        self.sam_model_id = sam_model_id
        self.dino_model_id = dino_model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Processors and models
        # ==================================================
        # GroundingDINO model.
        self.dino_processor = transformers.AutoProcessor.from_pretrained(
            self.dino_model_id,
            backend="torchvision",
        )
        self.dino_model = (
            transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                self.dino_model_id
            ).to(self.device)
        )
        self.dino_model.eval()

        # SAM2 model.
        self.sam_processor = transformers.Sam2Processor.from_pretrained(
            self.sam_model_id
        )
        self.sam_model = transformers.Sam2Model.from_pretrained(self.sam_model_id).to(
            self.device  # type: ignore[arg-type]
        )
        self.sam_model.eval()

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
        import torch

        # Flatten the label dictionary
        _prompt = utils.extract_categories(prompt)

        # Detect objects with GroundingDINO
        # ==================================================
        segmentations = []

        for idx, (uid, image) in tqdm(enumerate(zip(uids, images)), total=len(images)):
            # Dictionary that will hold all the information about the segmentation
            segmentation = {"uid": uid}

            # # Detect objects with GroundingDINO
            # ==================================================
            dino_inputs = self.dino_processor(
                images=[image],
                text=_prompt,
                return_tensors="pt",
            ).to(self.device)

            # Run the model on the processed images
            with torch.no_grad():
                dino_outputs = self.dino_model(**dino_inputs)

            # Process the results to detect objects and bounding boxes
            dino_results = self.dino_processor.post_process_grounded_object_detection(
                dino_outputs,
                dino_inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.shape[:2]],
            )[0]

            bboxes = dino_results["boxes"]
            if bboxes.numel() == 0 or bboxes.size()[0] == 0:
                # No objects found, but still record the image as processed.
                logger.debug(f"No objects found in image '{uid}'.")
                segmentation["labels"] = []
                segmentation["instances"] = np.zeros(
                    (0, *image.shape[:2]), dtype=np.bool_
                )
                segmentations.append(segmentation)
                continue

            # Bounding boxes
            bboxes = bboxes.cpu().numpy()

            # Segment the objects with SAM
            # ==================================================
            # Use SAM to segment objects based on bounding boxes.
            sam_inputs = self.sam_processor(
                images=[Image.fromarray(image.astype(np.uint8)).convert("RGB")],
                input_boxes=[bboxes],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs, multimask_output=False)

            # Process the model outputs
            masks = self.sam_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"],
            )[0]

            # Extract the instance-level object masks
            sam_masks = masks.numpy().squeeze(1)

            # Instance labels from GroundingDINO
            instance_labels = dino_results["text_labels"]

            # Populate the instance masks.
            instances = np.zeros(
                (len(instance_labels), *image.shape[:2]),
                dtype=np.bool_,
            )
            for mask_idx, sam_mask in enumerate(sam_masks):
                instances[mask_idx][sam_mask > 0] = True

            # Extract and store the segmentations.
            segmentation["labels"] = instance_labels
            segmentation["instances"] = instances
            segmentations.append(segmentation)

        return segmentations
