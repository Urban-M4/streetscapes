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
from tqdm import tqdm

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
        self._load_models()

    def _load_models(self):
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

    def _generate_sam_masks(self, image_rgb: np.ndarray) -> list[dict]:
        """

        Use the SAM mask generator to create a list of masks:

        segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        area - [int] - the area of the mask in pixels
        bbox - [List[int]] - the boundary box of the mask in xywh format
        predicted_iou - [float] - the model's own prediction for the quality of the mask
        point_coords - [List[List[float]]] - the sampled input point that generated this mask
        stability_score - [float] - an additional measure of mask quality
        crop_box - List[int] - the crop of the image used to generate this mask in xywh format
        """

        sam2_result = self.mask_generator.generate(image_rgb)
        return sam2_result

    def segment(
        self,
        images: ImageOrPath,
        labels: str | list[str],
        merge: bool = True,
    ) -> list[dict]:
        """
        Segment the provided sequence of images.

        Args:
            images (ImagePath):
                A list of images to process.

            labels (list[str]):
                Object categories to look for.

            merge (bool, optional):
                A switch indicating whether all instances of the same category
                should be merged into a single mask.
                Defaults to True.

        Returns:
            list[dict]:
                A list of dictionary objects containing instance-level segment information.
        """

        # Load the images as NumPy arrays
        images = self.load_images(images)
        image_names = list(images.keys())
        image_list = list(images.values())

        # Split the labels if they are provided together as one string.
        if isinstance(labels, str):
            labels = labels.split(".")
        labels = [f"{lbl.strip()}." for lbl in labels if len(lbl) > 0]

        # Run the GroundingDINO model
        # ==================================================
        logger.info("Detecting objects...")
        inputs = self.dino_processor(
            images=image_list,
            text=labels,
            return_tensors="pt",
        ).to(self.device)

        with pt.no_grad():
            outputs = self.dino_model(**inputs)

        dino_results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.shape[:2] for img in image_list],
        )

        # Perform the segmentation.
        # ==================================================
        logger.info("Segmenting patches...")

        results = []
        sam_indices = []
        sam_images = []
        sam_boxes = []
        sam_boxes = []
        for idx, result in enumerate(dino_results):

            if result["labels"]:
                sam_indices.append(idx)
                sam_images.append(image_list[idx])
                sam_boxes.append(result["boxes"].cpu().numpy())

        if sam_images:
            segmentations = self.segment_batch(sam_images, bboxes=sam_boxes)
            for idx, masks in zip(sam_indices, segmentations):

                labels = dino_results[idx]["labels"]
                labelled_masks = {}

                for label, mask in zip(labels, masks):
                    if label:
                        if merge:
                            labelled_masks.setdefault(
                                label, np.zeros_like(mask, dtype=np.bool)
                            )
                            labelled_masks[label][mask > 0] = True
                        else:
                            labelled_masks.setdefault(label, {})
                            labelled_masks[label]['instance'].append(mask)

                results.append(
                    {
                        "index": idx,
                        "name": image_names[idx],
                        "image": image_list[idx],
                        "categories": labelled_masks,
                    }
                )

                logger.info(
                    f"Extracted {len(masks)} masks for '<yellow>{image_names[idx]}</yellow>'"
                )
        return results

        # # --------------------------------------
        # with tqdm(total=len(images)) as pbar:
        #     for image_name, image in images.items():

        #             gdino_results = self.dino_model.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        #             all_results = []
        #             sam_images = []
        #             sam_boxes = []
        #             sam_indices = []
        #             for idx, result in enumerate(gdino_results):
        #                 processed_result = {
        #                     **result,
        #                     "masks": [],
        #                     "mask_scores": [],
        #                 }

        #                 if result["labels"]:
        #                     processed_result["boxes"] = result["boxes"].cpu().numpy()
        #                     processed_result["scores"] = result["scores"].cpu().numpy()
        #                     sam_images.append(np.asarray(images_pil[idx]))
        #                     sam_boxes.append(processed_result["boxes"])
        #                     sam_indices.append(idx)

        #                 all_results.append(processed_result)
        #             if sam_images:
        #                 print(f"Predicting {len(sam_boxes)} masks")
        #                 masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
        #                 for idx, mask, score in zip(sam_indices, masks, mask_scores):
        #                     all_results[idx].update(
        #                         {
        #                             "masks": mask,
        #                             "mask_scores": score,
        #                         }
        #                     )
        #                 print(f"Predicted {len(all_results)} masks")
        #             return all_results

        # --------------------------------------

        #             # Process the image with the processor
        #             inputs = self.processor(images=image, return_tensors="pt")
        #             inputs.to(self.device)
        #             pixel_values = inputs["pixel_values"].to(self.device)
        #             pixel_mask = inputs["pixel_mask"].to(self.device)

        #             # Pass the pixel masks through the model to obtain the segmentation.
        #             output = self.sam_model(
        #                 pixel_values=pixel_values, pixel_mask=pixel_mask
        #             )

        #         # Extract instance-level label information.
        #         # ==================================================
        #         # The `segmented` object is a dictionary containing the following items:
        #         # - 'segmentation':
        #         #   A tensor containing pixel-level instance IDs.
        #         # - 'segments_info':
        #         #   A list of dictionary objects with metadata about
        #         #   each instance, including its label ID.
        #         segmented = self.processor.post_process_panoptic_segmentation(
        #             output,
        #             threshold=self.threshold,
        #             mask_threshold=self.mask_threshold,
        #             overlap_mask_area_threshold=self.overlap_mask_area_threshold,
        #             label_ids_to_fuse=self.label_ids_to_fuse,
        #             target_sizes=[image.shape[:2]],
        #         )[0]

        #         # Extract the segmentation and the instance metadata
        #         segmentation = segmented["segmentation"]

        #         # Pop the segment metadata.
        #         segments = segmented.pop("segments_info")

        #         instances_per_label = {label_id: [] for label_id in self.labels}
        #         for segment in segments:

        #             instances_per_label[segment["label_id"]].append(
        #                 {
        #                     "mask": (segmentation == segment["id"]).numpy(),
        #                     "id": segment["id"],
        #                     "fused": segment["was_fused"],
        #                     "score": segment["score"],
        #                 }
        #             )

        #         # Populate the result dictionary
        #         # ==================================================
        #         results[image_name] = {
        #             "segmentation": segmentation,
        #             "labels": instances_per_label,
        #         }

        #         pbar.update()

        # return results

    def segment_single(
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

    def segment_batch(
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
