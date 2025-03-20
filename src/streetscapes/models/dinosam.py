# --------------------------------------
import typing as tp

# --------------------------------------
import torch as pt

# --------------------------------------
import numpy as np

# --------------------------------------
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --------------------------------------
from transformers import AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor

# --------------------------------------
import skimage as ski

# --------------------------------------
from streetscapes import conf
from streetscapes.conf import logger
from streetscapes.models import BaseSegmenter
from streetscapes.models import ImagePath


class DinoSAM(BaseSegmenter):

    def __init__(
        self,
        sam_model_id: str = "facebook/sam2.1-hiera-large",
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        *args,
        **kwargs,
    ):
        """
        A model combining SAM2 and GroundingDINO for promptable instance segmentation.
        Inspired by [LangSAM](https://github.com/luca-medeiros/lang-segment-anything) and [SamGeo](https://samgeo.gishub.org/samgeo/).

        Args:
            sam_model_id (str, optional):
                SAM2 model.
                Possible options include 'facebook/sam2.1-hiera-tiny', 'facebook/sam2.1-hiera-small' and 'facebook/sam2.1-hiera-large'.
                Defaults to 'sam2.1-hiera-large'.

            dino_model_id (str, optional):
                A GroundingDINO model.
                Defaults to "IDEA-Research/grounding-dino-base"

            box_threshold (float, optional):
                This parameter is used for modulating the identification of objects in the images.
                The box threshold is related to the model confidence,
                so a higher value makes the model more selective because
                it is equivalent to requiring the model to only select
                objects that it feels confident about.
                Defaults to 0.3.

            text_threshold:
                This parameter is also used for influencing the selectivity of the model,
                by requiring a stronger association between the prompt and the segment.
                Defaults to 0.3.
        """

        # Initialise the base
        super().__init__(*args, **kwargs)

        # Arguments
        # ==================================================
        self.sam_model_id = sam_model_id
        self.dino_model_id = dino_model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Processors and models
        # ==================================================
        self.sam_model: SAM2ImagePredictor = None
        self.sam_mask_generator: SAM2AutomaticMaskGenerator = None
        self.dino_processor: AutoProcessor = None
        self.dino_model: AutoModelForZeroShotObjectDetection = None
        self._load()

    def _load(self):
        """
        Convenience method for loading processors and models.
        """

        # SAM2 model.
        # ==================================================
        # Thre is no image processor for SAM.
        self.sam_model = SAM2ImagePredictor.from_pretrained(
            self.sam_model_id, device=self.device
        )

        # GroundingDINO model.
        # ==================================================
        self.dino_processor = AutoProcessor.from_pretrained(
            self.dino_model_id, device=self.device
        )
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_id
        ).to(self.device)
        self.dino_model.eval()

    def _merge_masks(
        self,
        image: np.ndarray,
        instance_masks: dict,
    ) -> dict[str, tp.Any]:

        # A global mask.
        # All instances will be accessible via this mask.
        global_masks = np.zeros(image.shape[:2], dtype=np.uint32)
        global_outlines = np.zeros_like(global_masks)

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

    def segment(
        self,
        images: ImagePath,
        labels: dict,
    ) -> list[dict]:
        """
        Segment the provided sequence of images.

        Args:
            images (ImagePath):
                A list of images to process.

            labels (dict):
                A flattened set of labels to look for,
                with optional subsets of labels that should be
                checked in order to eliminate overlaps.
                Cf. `BaseSegmenter._flatten_labels()`

        Returns:
            list[dict]:
                A list of dictionary objects containing
                instance-level segment information.
        """

        # Load the images as NumPy arrays
        image_paths, image_list = self.load_images(images)

        # Flatten the label dictionary
        labels = self._flatten_labels(labels)

        # Split the prompt if it is provided as a single string.
        prompt = " ".join([f"{lbl.strip()}." for lbl in labels if len(lbl) > 0])

        # Detect objects with GroundingDINO
        # ==================================================

        masks = {}
        instances = {}
        image_map = {}

        for idx, image in enumerate(image_list):
            orig_id = int(image_paths[idx].stem)
            image_map[orig_id] = image

            logger.info(f"[ <yellow>{orig_id}</yellow> ] Detecting objects...")

            inputs = self.dino_processor(
                images=[image],
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

            # Run the model on the input images
            with pt.no_grad():
                outputs = self.dino_model(**inputs)

            # Process the results to detect objects and bounding boxes
            dino_results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.shape[:2]],
            )[0]

            if not dino_results["labels"]:
                # No objects found, move on...
                continue

            # Bounding boxes
            bboxes = dino_results["boxes"].cpu().numpy()

            # Segment the objects with SAM
            # ==================================================
            logger.info(f"[ <yellow>{orig_id}</yellow> ] Performing segmentation...")

            # Use SAM to segment any images that contain objects.
            sam_masks = self._segment_single(image, bboxes=bboxes)

            # Labels from GroundingDINO
            dino_labels = dino_results["labels"]

            # Label to instance IDs
            insts_by_label = {}

            # Dictionary of final masks
            inst_masks = {}

            # A new dictionary of instances for this image
            instances[orig_id] = {}
            for inst_label, sam_mask in zip(dino_labels, sam_masks):
                _inst_id = len(inst_masks) + 1
                instances[orig_id][_inst_id] = inst_label
                insts_by_label.setdefault(inst_label, []).append(_inst_id)
                inst_masks[_inst_id] = sam_mask

            # Remove overlaps between labelled masks.
            logger.info(f"[ <yellow>{orig_id}</yellow> ] Removing overlaps...")

            # Computer and store the mask
            mask = self._remove_overlaps(image, inst_masks, insts_by_label, labels)
            masks[orig_id] = mask

            logger.info(
                f"[ <yellow>{orig_id}</yellow> ] Extracted {len(inst_masks)} instances for {len(insts_by_label)} labels."
            )

        return image_map, masks, instances

    def _segment_single(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
    ) -> np.ndarray:
        """
        Segment a single image.

        Args:
            image (np.ndarray):
                Image as a NumPy array.

            bbox (np.ndarray):
                A bounding box in XYXY format.

        Returns:
            np.ndarray:
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
        """
        Segment a batch of images.

        Args:
            images (list[np.ndarray]):
                Images to process.

            bboxes (list[np.ndarray]):
                Bounding boxes for all images in XYXY format.

        Returns:
            list[np.ndarray]:
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

    def _remove_overlaps(
        self,
        image: np.ndarray,
        masks: dict[int, np.ndarray],
        instances: dict[str, list[int]],
        labels: dict[str, list[str] | None],
    ) -> np.ndarray:
        """
        Remove overap between masks based on the specified label hierarchy.

        Args:
            image (np.ndarray):
                The image being segmented.

            masks (dict[int, np.ndarray]):
                A dictionary of instances and their coordinates.

            insstances (dict[str, list[int]]):
                A dictionary of labels mapped to instances of that label.

            labels (dict[str, list[str] | None]):
                A dictionary of labels and their dependencies
                that should be removed from the corresponding masks.
                (cf. `BaseSegmenter.flatten_labels()`).

        Returns:
            np.ndarray:
                A mask representing all segmented instances.
        """

        # A dictionary of merged masks for each label.
        label_masks = {}

        # Filtered instance masks
        filtered_instances = {}

        for label, inst_ids in instances.items():
            label_mask = np.zeros(image.shape[:2], dtype=bool)
            for inst_id in inst_ids:

                # Merge the instance
                label_mask |= masks[inst_id] > 0

                # Store the instance with its label.
                # This is used below for creating the global mask.
                filtered_instances[inst_id] = [label, masks[inst_id]]

            # Store the merged mask for this label
            label_masks[label] = label_mask

        # Iterate over all merged masks and remove overlaps.
        # and merge all masks into the global mask.
        # ==================================================

        # The final mask.
        # All instances will be accessible via this mask.
        mask = np.zeros(image.shape[:2], dtype=np.uint32)
        for inst_id, (label, inst_mask) in filtered_instances.items():

            # Remove any overlapping nested categories ("dependencies").
            if dep_labels := labels.get(label):
                for dep_label in dep_labels:
                    if dep_label in label_masks:
                        inst_mask[label_masks[dep_label]] = 0

            # Merge the instance into the global mask.
            mask[inst_mask > 0] = inst_id

        return mask
