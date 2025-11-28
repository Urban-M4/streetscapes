from cyclopts import App

from streetscapes.models import maskformer


segment_images_cli = App(help="Segment images")


@segment_images_cli.command(name="maskformer")
def segment_images_maskformer(
    image_path: str,
    labels: dict | None = None,
    batch_size: int = 10,
    params: dict = None,
    overwrite: bool = False,
):
    '''
    Segment images with the MaskFormer model.

    Args:
        image_path: Path to the images to be segmented.
        labels: Labels to focus on.
        batch_size: Batch size for the segmentation model.
        params: Model parameters.
        overwrite: Overwrite existing segmentations.
    '''


    maskformer.segment_images(image_path, labels, batch_size, params, overwrite)
