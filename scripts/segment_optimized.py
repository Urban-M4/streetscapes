#!/home/pkalverla1/Urban-M4/streetscapes/.venv/bin/python

import concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import threading
import queue
import torch
import transformers as tform
import sam2.sam2_image_predictor as sam2_pred

# ----- CONFIGURATION -----
BATCH_SIZE = 16  # Adjust based on your GPU memory
MAX_WORKERS = 16  # For CPU-bound tasks like image loading and saving
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

# ----- HELPER FUNCTIONS -----

def load_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    image = np.array(Image.open(path))
    original_shape = image.shape[:2]
    padded, _ = pad_right_bottom(image, target_size=2048)
    return padded, original_shape


def pad_right_bottom(img: np.ndarray, target_size: int = 2048):
    """Pad image on right and bottom to target_size.
    Returns padded image and (pad_h, pad_w) tuple for offset info."""
    h, w = img.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    if img.ndim == 3:
        padded = np.pad(
            img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
        )
    else:
        padded = np.pad(
            img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    return padded, (pad_h, pad_w)


def unpad_image(img: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Crop padded image back to original height and width."""
    h, w = original_shape
    return img[:h, :w]

def load_images_threaded(image_paths: list[Path], max_workers=MAX_WORKERS):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_image, image_paths))

    images, original_shapes = zip(*results)
    return list(images), list(original_shapes)


def save_segmentation_async(save_queue: queue.Queue):
    while True:
        item = save_queue.get()
        if item is None:
            break
        segmentation, save_path = item

        to_save = {
            "image_name": segmentation["image_path"].name,
            "masks": list(segmentation["masks"].items()),
            "instances": list(segmentation["instances"].items()),
        }

        np.savez_compressed(save_path, to_save)
        save_queue.task_done()


# ----- MODEL WRAPPER CLASS -----


class DinoSAMSegmenter:
    def __init__(
        self,
        sam_model_id="facebook/sam2.1-hiera-large",
        dino_model_id="IDEA-Research/grounding-dino-base",
        device=DEVICE,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Load models/processors
        self.sam_model = sam2_pred.SAM2ImagePredictor.from_pretrained(
            sam_model_id, device=device
        )
        self.dino_processor = tform.AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = tform.AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_id
        ).to(device)
        self.dino_model.eval()

    def detect_objects_batch(self, images: list[np.ndarray], prompt: str):
        """Run GroundingDINO on a batch of images."""
        inputs = self.dino_processor(
            images=images, text=[prompt] * len(images), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.shape[:2] for img in images],
        )
        return results

    def segment_batch(self, images: list[np.ndarray], bboxes_batch: list[np.ndarray]):
        """
        Segment images with SAM, skipping images with zero boxes to avoid token dimension mismatch.
        Returns masks_batch aligned with original images; for images without boxes, returns empty arrays.
        """

        # Filter images and boxes to only those with at least one box
        filtered_images = []
        filtered_boxes = []
        filtered_indices = []
        for idx, boxes in enumerate(bboxes_batch):
            if boxes.shape[0] > 0:
                filtered_images.append(images[idx])
                filtered_boxes.append(boxes)
                filtered_indices.append(idx)

        # Initialize masks list with None or empty masks
        masks_batch = [np.empty((0, *images[0].shape[:2]), dtype=bool) for _ in images]

        if len(filtered_images) == 0:
            # No boxes in entire batch, return empty masks
            return masks_batch

        # Run SAM segmentation only on filtered images
        self.sam_model.set_image_batch(filtered_images)
        filtered_masks_batch, _, _ = self.sam_model.predict_batch(
            box_batch=filtered_boxes, multimask_output=False
        )

        # Process masks shape: squeeze if needed
        filtered_masks_batch = [
            np.squeeze(masks, axis=1) if len(masks.shape) > 3 else masks
            for masks in filtered_masks_batch
        ]

        # Insert filtered masks back into masks_batch list at correct indices
        for idx, masks in zip(filtered_indices, filtered_masks_batch):
            masks_batch[idx] = masks

        return masks_batch


# ----- MAIN FLOW -----


def flatten_labels(labels: dict) -> dict:
    """Flatten label dict, similar to your original _flatten_labels."""

    def _flatten(tree, subtree=None):
        if subtree is None:
            subtree = {}
        for k, v in tree.items():
            if isinstance(v, dict):
                subtree[k] = list(v.keys())
                _flatten(v, subtree)
            else:
                subtree[k] = []
                if v is not None:
                    subtree[v] = []
                    subtree[k].append(v)
        return subtree

    return _flatten(labels)


def process_images(image_paths: list[Path], labels: dict, batch_size=BATCH_SIZE):
    segmenter = DinoSAMSegmenter()

    flat_labels = flatten_labels(labels)
    prompt = " ".join([lbl.strip() + "." for lbl in flat_labels if lbl])

    save_queue = queue.Queue()
    save_thread = threading.Thread(
        target=save_segmentation_async, args=(save_queue,), daemon=True
    )
    save_thread.start()

    for batch_start in tqdm(
        range(0, len(image_paths), batch_size), desc="Processing batches"
    ):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        images, original_shapes = load_images_threaded(batch_paths)

        # Adjust bounding boxes for padding: boxes come relative to original image,
        # but SAM expects boxes relative to padded images
        # So, boxes need no adjustment if padding is only on bottom/right,
        # because top-left coords are unchanged. Just ensure target_sizes use padded shapes.

        # Detect objects
        dino_results = segmenter.detect_objects_batch(images, prompt)

        bboxes_batch = []
        for idx, results in enumerate(dino_results):
            boxes = (
                results["boxes"].cpu().numpy()
                if len(results["boxes"]) > 0
                else np.zeros((0, 4))
            )

            bboxes_batch.append(boxes)

        masks_batch = segmenter.segment_batch(images, bboxes_batch)

        for idx, (image_path, dino_res, masks) in enumerate(
            zip(batch_paths, dino_results, masks_batch)
        ):
            if len(dino_res["labels"]) == 0:
                continue

            instance_masks = {}
            instances = {}

            for inst_id, (label, mask) in enumerate(
                zip(dino_res["labels"], masks), start=1
            ):
                mask = unpad_image(mask, original_shapes[idx])
                instances[inst_id] = label
                instance_masks[inst_id] = mask > 0  # Boolean mask instead of np.where

            segmentation = {
                "image_path": image_path,
                "masks": instance_masks,
                "instances": instances,
            }

            save_dir = image_path.parent / "segmentations" / "dinosam"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / (image_path.stem + ".npz")

            save_queue.put((segmentation, save_path))

    save_queue.join()
    save_queue.put(None)
    save_thread.join()


# -------- USAGE --------
if __name__ == "__main__":
    # Example usage with paths and labels
    from streetscapes.utils import get_env

    image_dir = Path(get_env("DATA_HOME")) / "sources" / "mapillary" / "images"
    existing_images = list(image_dir.glob("*.jpeg"))

    labels = {
        "building": {"building": None, "window": None, "door": None},
        "road": {
            "road": None,
            "street": None,
            "sidewalk": None,
            "pavement": None,
            "crosswalk": None,
        },
        "vegetation": None,
        "car": None,
        "truck": None,
    }

    process_images(existing_images, labels, batch_size=BATCH_SIZE)
