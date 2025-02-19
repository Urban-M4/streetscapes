# --------------------------------------
import torch as pt

# --------------------------------------
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

# --------------------------------------
import skimage as ski

# --------------------------------------
import numpy as np

# --------------------------------------
from streetscapes import logger
from streetscapes.models import BaseSegmenter
from streetscapes.models import ImageOrPath


class MaskFormer(BaseSegmenter):

    # All the labels recognised by Mask2Former.
    id_to_label = {
        0: "bird",
        1: "ground-animal",
        2: "curb",
        3: "fence",
        4: "guard-rail",
        5: "barrier",
        6: "wall",
        7: "bike-lane",
        8: "crosswalk-plain",
        9: "curb-cut",
        10: "parking",
        11: "pedestrian-area",
        12: "rail-track",
        13: "road",
        14: "service-lane",
        15: "sidewalk",
        16: "bridge",
        17: "building",
        18: "tunnel",
        19: "person",
        20: "bicyclist",
        21: "motorcyclist",
        22: "other-rider",
        23: "lane-marking-crosswalk",
        24: "lane-marking-general",
        25: "mountain",
        26: "sand",
        27: "sky",
        28: "snow",
        29: "terrain",
        30: "vegetation",
        31: "water",
        32: "banner",
        33: "bench",
        34: "bike-rack",
        35: "billboard",
        36: "catch-basin",
        37: "cctv-camera",
        38: "fire-hydrant",
        39: "junction-box",
        40: "mailbox",
        41: "manhole",
        42: "phone-booth",
        43: "pothole",
        44: "street-light",
        45: "pole",
        46: "traffic-sign-frame",
        47: "utility-pole",
        48: "traffic-light",
        49: "traffic-sign-back",
        50: "traffic-sign-front",
        51: "trash-can",
        52: "bicycle",
        53: "boat",
        54: "bus",
        55: "car",
        56: "caravan",
        57: "motorcycle",
        58: "on-rails",
        59: "other-vehicle",
        60: "trailer",
        61: "truck",
        62: "wheeled-slow",
        63: "car-mount",
        64: "ego-vehicle",
    }

    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        labels_to_fuse: set[str | int] = None,
        *args,
        **kwargs,
    ):
        """
        A wrapper for the [Mask2Former model](https://huggingface.co/docs/transformers/en/model_doc/mask2former).

        The following documentation for the model parameters is taken from the HuggingFace
        page for the panoptic [processing pipeline](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_panoptic_segmentation)
        for the Mask2Former model.

        These parameters are passed directly to the corresponding arguments of the
        post_process_panoptic_segmentation() method of the image processor.

        Args:
            model_id (str, optional):
                Mask2Former model to load.
                Defaults to "facebook/mask2former-swin-large-mapillary-vistas-panoptic".

            threshold (float, optional):
                The probability score threshold to keep predicted instance masks.
                Defaults to 0.5.

            mask_threshold (float, optional):
                Threshold to use when turning the predicted masks into binary values.
                Defaults to 0.5.

            overlap_mask_area_threshold (float, optional):
                The overlap mask area threshold to merge or discard small disconnected
                parts within each binary instance mask.The overlap mask area threshold
                to merge or discard small disconnected parts within each binary instance mask.
                Defaults to 0.8.

            labels_to_fuse (list, optional):
                The labels in this state will have all their instances be fused together.
                For instance, we could say there can only be one sky in an image, but several
                persons, so the label ID for sky would be in that set, but not the one for person.
                This differs slightly from the original parameter because it can also accept
                strings instead of integers (the strings are converted to their IDs using the ).
                Defaults to None.
        """

        # Initialise the base
        super().__init__(*args, **kwargs)

        self.id_to_label = MaskFormer.id_to_label

        # Create the reverse mapping of label to label ID
        self.label_to_id = {
            label: label_id for label_id, label in self.id_to_label.items()
        }

        # Arguments
        # ==================================================
        # Convert any string labels into integers
        label_ids_to_fuse = set()
        if labels_to_fuse is not None:
            for lbl in labels_to_fuse:
                if isinstance(lbl, int):
                    label_ids_to_fuse.add(lbl)
                elif isinstance(lbl, str):
                    label_ids_to_fuse.add(self.label_to_id[lbl])

        self.model_id = model_id
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.overlap_mask_area_threshold = overlap_mask_area_threshold
        self.label_ids_to_fuse = label_ids_to_fuse

        # Processors and models
        # ==================================================
        self.processor: Mask2FormerImageProcessor = None
        self.model: Mask2FormerForUniversalSegmentation = None
        self._load()

    def _load(self):
        """
        Convenience method for loading processors and models.
        """

        # Mask2Former model
        # ==================================================
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            self.model_id,
            use_fast=True,
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

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

        # Eliminate labels that are not recognised by the model
        remove = set(labels).difference(self.id_to_label)
        _labels = {}
        for k, v in labels.items():
            if k in remove:
                continue
            vdiff = set(v) - remove
            _labels[k] = list(vdiff) if len(vdiff) > 0 else None
        labels = _labels

        results = {}
        logger.info("Segmenting images...")

        # Perform the segmentation.
        # ==================================================
        logger.info("Detecting objects...")
        with pt.no_grad():
            # Process the image with the processor
            inputs = self.processor(images=image_list, return_tensors="pt")
            inputs.to(self.device)
            pixel_values = inputs["pixel_values"].to(self.device)
            pixel_mask = inputs["pixel_mask"].to(self.device)

            # Pass the pixel masks through the model to obtain the segmentation.
            output = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            segmented = self.processor.post_process_panoptic_segmentation(
                output,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                overlap_mask_area_threshold=self.overlap_mask_area_threshold,
                label_ids_to_fuse=self.label_ids_to_fuse,
                target_sizes=[img.shape[:2] for img in image_list],
            )

            # Some containers for intermediate results that can be
            # accessed during the segmentation phase.
            results = []
            for idx, result in enumerate(segmented):

                # Sort out instances and their labels
                segmentation = result["segmentation"].detach().clone().cpu().numpy()
                instances = {}
                outlines = np.zeros_like(segmentation)
                for instance in result["segments_info"]:
                    inst_id = instance["id"]
                    inst_label = self.id_to_label[instance["label_id"]]
                    instances[inst_id] = inst_label

                    # Find the outline of the instance
                    outline = ski.segmentation.find_boundaries(
                        np.where(segmentation == inst_id, 1, 0), mode="thick"
                    )
                    outlines[outline] = inst_id

                stats = self.compute_stats(image_list[idx], segmentation, instances)

                results.append(
                    {
                        "masks": segmentation,
                        "outlines": outlines,
                        "stats": stats,
                    }
                )

                logger.info(
                    f"[ <yellow>{image_names[idx]}</yellow> ] Extracted {len(instances)} instances for {len(set(instances.values()))} labels."
                )

        return results
