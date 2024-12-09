# --------------------------------------
import torch as pt

# --------------------------------------
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
from streetscapes import conf
from streetscapes.conf import logger
from streetscapes.models import BaseSegmenter
from streetscapes.models import ImagePath


class MaskFormerVistasPanoptic(BaseSegmenter):

    # All the labels that the Mask2Former recognises.
    labels = {
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

        # Load the processor and the model
        self.processor, self.model = self.load_model()

        # Convert any string labels into integers
        label_ids_to_fuse = set()
        if labels_to_fuse is not None:
            for lbl in labels_to_fuse:
                if isinstance(lbl, int):
                    label_ids_to_fuse.add(lbl)
                elif isinstance(lbl, str):
                    label_ids_to_fuse.add(self.label_ids[lbl])

        # Panoptic pipeline options.
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.overlap_mask_area_threshold = overlap_mask_area_threshold
        self.label_ids_to_fuse = label_ids_to_fuse

    def load_model(
        self,
    ) -> tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation]:
        """
        Convenience function for loading the image processor
        and the segmentation model.

        Returns:
            tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation]:
                A tuple holding the image processor and the model.
        """

        # Mask2Former
        # ==================================================
        processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
        )

        model.to(self.device)
        return (processor, model)

    def segment(
        self,
        images: ImagePath,
    ) -> list[dict]:
        """
        Segment the provided sequence of images.

        Args:
            images (ImagePath):
                A list of images to process.

        Returns:
            list[dict]:
                A list of dictionary objects containing instance-level segment information.
        """

        # Load the images
        images = self.load_images(images)

        # Perform the segmentation.
        # ==================================================
        results = {}
        logger.info("Segmenting images...")

        with tqdm(total=len(images)) as pbar:
            for image_name, image in images.items():
                with pt.no_grad():
                    # Process the image with the processor
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs.to(self.device)
                    pixel_values = inputs["pixel_values"].to(self.device)
                    pixel_mask = inputs["pixel_mask"].to(self.device)

                    # Pass the pixel masks through the model to obtain the segmentation.
                    output = self.model(
                        pixel_values=pixel_values, pixel_mask=pixel_mask
                    )

                # Extract instance-level label information.
                # ==================================================
                # The `segmented` object is a dictionary containing the following items:
                # - 'segmentation':
                #   A tensor containing pixel-level instance IDs.
                # - 'segments_info':
                #   A list of dictionary objects with metadata about
                #   each instance, including its label ID.
                segmented = self.processor.post_process_panoptic_segmentation(
                    output,
                    threshold=self.threshold,
                    mask_threshold=self.mask_threshold,
                    overlap_mask_area_threshold=self.overlap_mask_area_threshold,
                    label_ids_to_fuse=self.label_ids_to_fuse,
                    target_sizes=[image.shape[:2]],
                )[0]

                # Extract the segmentation and the instance metadata
                segmentation = segmented["segmentation"]

                # Pop the segment metadata.
                segments = segmented.pop("segments_info")

                instances_per_label = {label_id: [] for label_id in self.labels}
                for segment in segments:

                    instances_per_label[segment["label_id"]].append(
                        {
                            "mask": (segmentation == segment["id"]).numpy(),
                            "id": segment["id"],
                            "fused": segment["was_fused"],
                            "score": segment["score"],
                        }
                    )

                # Populate the result dictionary
                # ==================================================
                results[image_name] = {
                    "segmentation": segmentation,
                    "labels": instances_per_label,
                }

            pbar.update()

        return results
