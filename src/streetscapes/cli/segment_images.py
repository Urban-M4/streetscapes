import json
import os
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from streetscapes.models.ade20k import ADE20KFacade
from streetscapes.models.groundingdino import GroundingDINO
from streetscapes.models.sam import SAM

segment_images_cli = typer.Typer(help="Segment images")


# ---------------------------
# Helpers
# ---------------------------
def _parse_image_input(images: list[str] | str) -> list[str]:
    """Normalize input to a list of image paths.

    - Single image path
    - Folder containing images
    - Comma-separated list of images
    - JSON manifest file with {"image": <path>} entries
    """
    if isinstance(images, str):
        images = [images]

    all_paths = []
    for img in images:
        p = Path(img)
        if p.is_dir():
            all_paths.extend(
                [str(f) for f in p.iterdir() if f.suffix.lower() in {".jpg", ".png"}]
            )
        elif p.suffix.lower() == ".json":
            # assume manifest
            with open(p) as f:
                data = json.load(f)
            all_paths.extend([entry["image"] for entry in data])
        elif "," in img:
            all_paths.extend(img.split(","))
        else:
            all_paths.append(str(p))
    return all_paths


def _load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _save_masks(masks, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{filename}_masks.npy"), masks)


# ---------------------------
# Standalone models that can operate directly on a list of images
# ---------------------------
@segment_images_cli.command("sam")
def segment_images_sam(images: str, out: str):
    """Segment images with SAM."""
    images_list = _parse_image_input(images)
    model = SAM(checkpoint="sam_vit_h_4b8939.pth")
    manifest = model.process_batch(images_list, out)
    typer.echo(f"SAM finished. Manifest: {manifest}")


@segment_images_cli.command("dino")
def segment_images_dino(images: list[str], out: str):
    """Detect objects with GroundingDino"""
    images_list = _parse_image_input(images)
    model = GroundingDINO(
        config="GroundingDINO_SwinT_OGC.py", weights="groundingdino_swint_ogc.pth"
    )
    manifest = model.process_batch(images_list, out)
    typer.echo(f"GroundingDINO finished. Manifest: {manifest}")


@segment_images_cli.command("openclip")
def classify_images_openclip(
    image_path: str,
    labels: str,  # comma-separated
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
):
    labels_list = labels.split(",")
    image = np.array(Image.open(image_path).convert("RGB"))
    model = OpenCLIP(model_name=model_name, pretrained=pretrained)
    label = model.classify(image, labels_list)
    typer.echo(f"Predicted label: {label}")


@segment_images_cli.command("clipseg")
def segment_images_clipseg(
    image_path: str,
    prompt: str,
    output_path: str = "./clipseg_mask.npy",
):
    image = np.array(Image.open(image_path).convert("RGB"))
    model = CLIPSeg()
    mask = model.predict(image, prompt)
    np.save(output_path, mask)
    typer.echo(f"Saved mask to {output_path}")


@segment_images_cli.command("ade20k")
def segment_images_ade20k(
    images: str,
    output_dir: str = "./output",
    encoder_weights: str = "ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth",
    decoder_weights: str = "ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth",
    building_min_fraction: float = 0.2,
):
    """Segment images with ADE20K.

    Returns masked images, empty mask, and manifest per image.
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ADE20KFacade(encoder_weights, decoder_weights, device)

    image_paths = _parse_image_input(images)

    for path in image_paths:
        filename = Path(path).stem
        try:
            img_np = _load_image(path)
            masked_rgb, empty_mask, manifest = model.segment(
                img_np, building_min_fraction
            )

            # Save masked image and masks
            masked_out_dir = os.path.join(output_dir, "masked")
            masks_out_dir = os.path.join(output_dir, "masks")
            os.makedirs(masked_out_dir, exist_ok=True)
            os.makedirs(masks_out_dir, exist_ok=True)

            Image.fromarray(masked_rgb).save(
                os.path.join(masked_out_dir, f"{filename}_masked.png")
            )
            _save_masks(manifest, masks_out_dir, filename)

            print(f"Processed {filename}")

        except Exception as e:
            print(f"Failed {filename}: {e}")


# ---------------------------
# Composite models chaining multiple models together and passing data between them.
# ---------------------------
@segment_images_cli.command("dinosam")
def segment_images_dinosam(
    images: str,  # list[str] | str ; typer doesn't support union
    out: str,
    prompt: str,
    sam_model: str = "facebook/sam2.1-hiera-large",
    dino_model: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    output_dir: str = "./output",
):
    """Detect with GroundingDino, then segment with SAM.

    Outputs masks and a manifest file with bounding boxes and labels.
    """
    image_paths = _parse_image_input(images)

    # Initialize models
    sam = SAM(model_id=sam_model)
    dino = GroundingDINO(model_id=dino_model)

    manifest = []

    for path in image_paths:
        img = _load_image(path)

        # Dino
        dino_result = dino.detect(
            img, prompt, box_threshold=box_threshold, text_threshold=text_threshold
        )
        boxes = dino_result["boxes"]
        labels = dino_result["labels"]

        # SAM
        sam_masks = sam.segment(img, boxes=boxes if len(boxes) > 0 else None)

        # Save masks
        out_base = Path(output_dir) / Path(path).stem
        _save_masks(sam_masks, output_dir, Path(path).stem)

        # Update manifest
        manifest.append(
            {
                "image": path,
                "boxes": boxes.tolist() if len(boxes) > 0 else [],
                "labels": labels,
                "mask_file": str(out_base) + "_masks.npy",
            }
        )

    # Save manifest
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    typer.echo(f"DinoSAM finished. Manifest: {manifest_path}")
