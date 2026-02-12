import typing as tp
import numpy as np
import skimage as ski
import uuid
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
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = utils.get_device(device)

        # Model parameters
        # ==================================================
        self.sam_model_id = sam_model_id
        self.dino_model_id = dino_model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Processors and models
        # ==================================================
        # SAM2 model.
        # Thre is no image processor for SAM.
        self.sam_model = SAM2ImagePredictor.from_pretrained(
            self.sam_model_id, device=self.device
        )

        # GroundingDINO model.
        self.dino_processor = transformers.AutoProcessor.from_pretrained(
            self.dino_model_id,
            use_fast=True,
        )
        self.dino_model = (
            transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                self.dino_model_id
            ).to(self.device)
        )
        self.dino_model.eval()

    # DEPRECATED
    def _merge_masks(
        self,
        image: np.ndarray,
        instance_masks: dict,
    ) -> dict[str, tp.Any]:
        """Merge separate instance masks.

        Args:
            image: The image being segmented.
            instance_masks: A dictionary of instance masks.

        Returns:
            A dictionary of merged masks.

        """
        # A global mask.
        # All instances will be accessible via this mask.
        global_masks = np.zeros(image.shape[:2], dtype=np.uint32)

        # A dictionary of merged masks for each label.
        merged_masks = {}

        # Mapping from instance ID to label
        instance_ids = {}

        # Mapping from label to instance ID
        label_to_instances = {}

        for label, instances in instance_masks.items():
            merged_mask = np.zeros_like(global_masks, dtype=bool)
            for instance in instances:
                # Merge the instance
                merged_mask |= instance > 0

                # Find the outline of the instance
                outline = ski.segmentation.find_boundaries(instance, mode="thick")

                # Instance ID
                inst_id = len(instance_ids) + 1

                # Store the instance with its label and outline.
                # This is used below for creating the global mask and outline maps.
                instance_ids[inst_id] = [label, instance, outline]

                # Create the reverse mapping (label -> list of instances)
                label_to_instances.setdefault(label, set()).add(inst_id)

            # Store the merged mask for this label
            merged_masks[label] = merged_mask

    def _segment_single(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
    ) -> np.ndarray:
        """Segment a single image.

        Args:
            image: Image as a NumPy array.
            bbox: A bounding box in XYXY format.

        Returns:
            A mask.

        """
        self.sam_model.set_image(image)
        masks, _, _ = self.sam_model.predict(box=bboxes, multimask_output=False)
        if len(masks.shape) > 3:
            masks = np.squeeze(masks, axis=1)
        return masks

    def _segment_batch(
        self,
        images: list[np.ndarray],
        bboxes: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Segment a batch of images.

        Args:
            images: Images to process.
            bboxes: Bounding boxes for all images in XYXY format.

        Returns:
            A list of masks.

        """
        self.sam_model.set_image_batch(images)

        masks, _, _ = self.sam_model.predict_batch(
            box_batch=bboxes, multimask_output=False
        )

        masks = [
            np.squeeze(mask, axis=1) if len(mask.shape) > 3 else mask for mask in masks
        ]
        return masks

    def segment_images(
        self,
        uids: list[uuid.UUID],
        images: list[np.ndarray],
        prompt: dict,
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
        prompt = utils.extract_categories(prompt)

        # Detect objects with GroundingDINO
        # ==================================================
        segmentations = []

        with torch.no_grad():

            for idx, (uid, image) in tqdm(enumerate(zip(uids, images)), total=len(images)):
                # Dictionary that will hold all the information about the segmentation
                segmentation = {"uid": uid}

                # Detect objects
                inputs = self.dino_processor(
                    images=[image],
                    text=prompt,
                    return_tensors="pt",
                ).to(self.device)

                # Run the model on the processed images
                with torch.no_grad():
                    outputs = self.dino_model(**inputs)

                # Process the results to detect objects and bounding boxes
                dino_results = (
                    self.dino_processor.post_process_grounded_object_detection(
                        outputs,
                        inputs["input_ids"],
                        threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=[image.shape[:2]],
                    )[0]
                )

                if not dino_results["text_labels"]:
                    # No objects found, move on...
                    logger.debug(f"Image")
                    continue

                # Bounding boxes
                bboxes = dino_results["boxes"].cpu().numpy()

                # Segment the objects with SAM
                # ==================================================
                # Use SAM to segment any images that contain objects.
                sam_masks = self._segment_single(image.astype(np.float32), bboxes=bboxes)

                # Instance labels from GroundingDINO
                instance_labels = dino_results["text_labels"]

                # A new dictionary of instances for this image
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
