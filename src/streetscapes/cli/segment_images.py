from pathlib import Path
import filetype as ft

from cyclopts import App

from streetscapes.models import maskformer


segment_images_cli = App(help="Segment images")


@segment_images_cli.command(name="maskformer")
def segment_images_maskformer(
    image_path: str,
    labels: list[str] | None = None,
    batch_size: int = 10,
    model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_threshold: float = 0.8,
    fuse_labels: list[str] | None = None,
    overwrite: bool = False,
    project: str = "streetscapes",
):
    """
    Segment images with the MaskFormer model.

    Args:
        image_path: Path to the images to be segmented.
        labels: Labels to focus on.
        batch_size: Batch size for the segmentation model.
        model_id: Mask2Former model to load.
        threshold: The probability score threshold to keep predicted instance masks.
        mask_threshold: Threshold to use when turning the predicted masks into binary values.
        overlap_threshold: The overlap mask area threshold to merge or discard small
            disconnected parts within each binary instance mask.
        fuse_labels: The labels in this state will have all their instances be fused together.
        overwrite: Overwrite existing segmentations.
        project: Project to save to
    """

    if fuse_labels is None or len(fuse_labels) < 2:
        # Fusing a single label makes no sense...
        fuse_labels = []

    model_params = {
        "model_id": model_id,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "overlap_mask_area_threshold": overlap_threshold,
        "labels_to_fuse": fuse_labels,
    }

    if image_path is not None:
        image_path = Path(image_path)

    if image_path.is_dir():
        image_path = [im_path for im_path in image_path.glob("*.*") if ft.is_image(im_path)]

    maskformer.segment_images(
        image_path, labels, batch_size, model_params, overwrite, project
    )
