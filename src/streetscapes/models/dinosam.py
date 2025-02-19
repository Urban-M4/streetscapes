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
from streetscapes.models import ImageOrPath


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

        # SAM model.
        # ==================================================
        # Thre is no image processor for SAM.
        self.sam_model = SAM2ImagePredictor.from_pretrained(
            self.sam_model_id, device=self.device
        )

        # GroundingDINO model.
        # ==================================================
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_id
        )
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
        images: ImageOrPath,
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
        images = self.load_images(images)
        image_names = list(images.keys())
        image_list = list(images.values())

        # Flatten the label dictionary
        labels = self._flatten_labels(labels)

        # Split the prompt if it is provided as a single string.
        prompt = " ".join([f"{lbl.strip()}." for lbl in labels if len(lbl) > 0])

        # Detect objects with GroundingDINO
        # ==================================================
        logger.info("Detecting objects...")
        inputs = self.dino_processor(
            images=image_list,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        # Run the model on the input images
        with pt.no_grad():
            outputs = self.dino_model(**inputs)

        # Process the results to detect objects and bounding boxes
        dino_results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.shape[:2] for img in image_list],
        )

        # Some containers for intermediate results that can be
        # accessed during the segmentation phase.
        results = []
        seq_indices = []
        seq_images = []
        seq_boxes = []
        for idx, result in enumerate(dino_results):

            if result["labels"]:
                seq_indices.append(idx)
                seq_images.append(image_list[idx])
                seq_boxes.append(result["boxes"].cpu().numpy())

        # Segment the objects with SAM
        # ==================================================
        logger.info("Performing segmentation...")

        if seq_images:

            # Use SAM to segment any images that contain objects.
            segmentations = self._segment_batch(seq_images, bboxes=seq_boxes)
            for idx, segmentation in zip(seq_indices, segmentations):

                # Extract the labels
                dino_labels = dino_results[idx]["labels"]

                # Dictionary of instances
                masks = {}

                for inst_label, instances in zip(dino_labels, segmentation):
                    if inst_label:
                        masks.setdefault(inst_label, []).append(instances)

                # Remove any overlaps between labeled masks.
                masks, outlines, instances = self._remove_overlaps(
                    image_list[idx], labels, masks
                )

                logger.info(
                    f"[ <yellow>{image_names[idx]}</yellow> ] Extracted {len(instances)} instances for {len(set(instances.values()))} labels."
                )

                stats = self.compute_stats(image_list[idx], masks, instances)

                results.append(
                    {
                        "masks": masks,
                        "outlines": outlines,
                        "stats": stats,
                    }
                )

        return results

    def _segment_single(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
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
        masks, _, _ = self.sam_model.predict(box=bbox, multimask_output=False)
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
        labels: dict[str, list[str] | None],
        masks: dict[str, list[np.ndarray]],
    ) -> tuple[np.ndarray, dict, dict]:
        """
        Remove overap between masks based on the specified label hierarchy.

        Args:
            image (np.ndarray):
                The image being segmented.

            labels (dict[str, list[str] | None]):
                Hierarchical label specification.
                (cf. `BaseSegmenter.flatten_labels()`).

            masks (dict[str, list[np.ndarray]]):
                A dictionary of labels and their corresponding instances.

        Returns:
            tuple[np.ndarray, dict, dict]:
                A tuple containing:
                    1. A global mask representing all segmented instances.
                    2. A forward mapping from instances to their labels.
                    3. A reverse mapping from labels to their instances.
        """

        logger.info(f"Removing overlaps...")

        # A global mask.
        # All instances will be accessible via this mask.
        global_masks = np.zeros(image.shape[:2], dtype=np.uint32)
        global_outlines = np.zeros_like(global_masks)
        global_instances = {}

        # A dictionary of merged masks for each label.
        merged_masks = {}

        for label, instances in masks.items():
            merged_mask = np.zeros_like(global_masks, dtype=bool)
            for instance in instances:

                # Merge the instance
                merged_mask |= instance > 0

                # Find the outline of the instance
                outline = ski.segmentation.find_boundaries(instance, mode="thick")

                # Instance ID
                inst_id = len(global_instances) + 1

                # Store the instance with its label and outline.
                # This is used below for creating the global mask and outline maps.
                global_instances[inst_id] = [label, instance, outline]

            # Store the merged mask for this label
            merged_masks[label] = merged_mask

        # Iterate over all merged masks and remove overlaps.
        # and merge all masks into the global mask.
        for instance_id, (label, instance, outline) in global_instances.items():

            # Remove any overlapping nested categories.
            if subcats := labels.get(label):
                for subcat in subcats:
                    if subcat in merged_masks:
                        instance[merged_masks[subcat]] = 0

            # Merge the instance into the global mask.
            global_masks[instance > 0] = instance_id

            # Merge the instance *outline* into the global outline map.
            global_outlines[outline > 0] = instance_id

            # Remove the instance from the instance dictionary,
            # it is not needed any more.
            global_instances[instance_id] = label

        return (global_masks, global_outlines, global_instances)
