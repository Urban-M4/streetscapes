# --------------------------------------
import re

# --------------------------------------
from pathlib import Path

# --------------------------------------
from PIL import Image
from PIL import ImageFile

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import numpy as np

# --------------------------------------
import scipy

# --------------------------------------
import torch as pt

# --------------------------------------
import skimage as ski

# --------------------------------------
import pandas as pd

# --------------------------------------
import awkward as ak

# --------------------------------------
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from streetscapes import conf
from streetscapes.conf import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

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

def extract_instances(segmentation: dict) -> dict:
    """
    Extract individual instances of objects in the recognised categories.

    Args:
        segmentation (dict):
            A dictionary containing an instance map (a tensor)
            and a list of instance / label mappings.

    Returns:
        dict:
            A dictionary of instances and their corresponding masks.
    """

    instance_map, segments = (
        segmentation["segmentation"].int(),
        segmentation["segments_info"],
    )

    (list_unique, list_counts) = pt.unique(instance_map, return_counts=True)

    if -1 in list_unique:
        list_unique = list_unique[1:]
        list_counts = list_counts[1:]

    instances = {label_id: [] for label_id in labels}
    for i, instance in enumerate(segments):
        instances[instance["label_id"]].append(
            {
                "mask": instance_map == instance["id"],
                "id": instance["id"],
            }
        )

    return instances


def segment_image(
    image: Image,
    processor: AutoImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
    threshold: float = 0.5,
    instance: bool = True,
) -> pt.Tensor:
    """
    Segment a single image.

    Args:
        image (Image):
            The image being segmented.

        processor (AutoImageProcessor):
            Image processor for feature extraciton.

        model (Mask2FormerForUniversalSegmentation):
            Segmentation model.

        threshold (float, optional):
            Threshold for keeping instances separate.
            Defaults to 0.5.

        instance (bool, optional):
            Toggle between instance and semantic segmentation mode.

    Returns:
        pt.Tensor:
            A 2D tensor of the same size as the image
            holding per-pixel class information.
    """

    with pt.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs.to(device)
        pixel_values = inputs["pixel_values"].to(device)
        pixel_mask = inputs["pixel_mask"].to(device)
        output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if instance:
            segmentation = processor.post_process_instance_segmentation(
                output,
                target_sizes=[(image.height, image.width)],
                threshold=threshold,
            )[0]

            instances = extract_instances(segmentation)

        else:
            segmentation = processor.post_process_semantic_segmentation(
                output,
                target_sizes=[(image.height, image.width)],
            )[0]

    return instances


def mask2former_vistas_panoptic() -> (
    tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation]
):
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

    model.to(device)
    return (processor, model)


def colour_dict(
    categories: dict,
    opacity: int = 255,
    cmap: str = "jet",
) -> dict:
    """
    Dictionary of colours (used for visualising the segments).

    Args:
        categories (tp.Dict):
            A dictionary of categories.

        opacity (int, optional):
            Opacity to use for the segmentation patches. Defaults to 255.

        cmap (str, optional):
            Matplotlib colourmap. Defaults to "jet".

    Returns:
        tp.Dict:
            Dictionary of class/colour associations.
    """
    cmap = plt.get_cmap(cmap)
    colours = {k: None for k in categories}
    n_labels = len(categories)
    for i, (cls, name) in enumerate(categories.items()):
        colour = np.array(cmap(1.0 * i / (n_labels)))
        colour[-1] = 1
        colours[cls] = opacity * colour
    return colours


def compute_stats(
    image: pt.Tensor,
    instances: dict,
):
    """
    Compute the statistics for all categories.

    Args:
        image (pt.Tensor):
            The image being processed.

        instances (dict):
            A dictionary mapping of label IDs to a list of instances.
    """

    image = np.array(image)

    results = {labels[label_id]: [] for label_id in instances}
    for label_id, instance_list in instances.items():

        for instance in instance_list:

            mask = instance["mask"]
            masked = image[mask]

            results[labels[label_id]].append(
                {
                    "colour": {
                        "median": np.median(masked, axis=0).astype(np.ubyte),
                        "mode": scipy.stats.mode(masked, axis=0)[0].astype(np.ubyte),
                        "mean": np.mean(masked, axis=0).astype(np.ubyte),
                        "sd": np.std(masked, axis=0).astype(np.ubyte),
                    },
                    # Area
                    "area": mask.count_nonzero().item() / np.prod(image.shape[:2]),
                }
            )

    return results


def visualise_segmentation(
    image_path: Path | str,
    categories: list = None,
    opacity: float = 0.50,
    figsize: tuple[int, int] = (16, 8),
    stats: ak.Array = None,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Segment and visualise an image, and potentially show the colour statistics
    associated with the requested categories.

    Args:
        image_path (tp.Union[Path, str]):
            Path to the image being visualised.

        categories (tp.List, optional):
            Categories to extract. Defaults to None.

        opacity (float, optional):
            Opacity to use for the segmentation overlay. Defaults to 0.50.

        figsize (tp.Tuple[int, int], optional):
            Figure size. Defaults to (16, 8).

        stats (ak.Array, optional):
            An `awkward` array of statistics. Defaults to None.

    Returns:
        tp.Tuple[plt.Figure, plt.Axes, tp.Dict]:
            A tuple containing:
                - A Figure object;
                - An Axes object;
                - A dictionary of colour statistics for the requested categories.
                    This will be an empty dictionary if `ds` is not provided.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"The provided path f'{image_path.relative_to(conf.ROOT_DIR)}' does not exist."
        )

    # Load the model and segment the image.
    # ==================================================
    (processor, model) = mask2former_vistas_panoptic()

    image = Image.open(image_path)
    instances = segment_image(image, processor, model)

    # Unify the styles of labels and categories
    # so that they can be compared.
    # ==================================================
    temp_categories = set()
    categories_to_ids = {cat.lower(): cat_id for cat_id, cat in labels.items()}
    if categories is None:
        categories = labels.values()

    categories = set(categories)

    for cat in categories:
        cat = re.sub(r"\s+", "-", cat.lower().strip().replace("-", " "))
        if cat in categories_to_ids:
            temp_categories.add(cat)

    categories = temp_categories
    category_dict = {categories_to_ids[cat]: cat for cat in categories}

    # Prepare the colour dictionary and the layers
    # necessary for plotting the category patches.
    # ==================================================
    image = np.array(image)
    image_opacity = 1 - opacity
    opacity_channel = np.full_like(image[:, :, 0], 255)[:, :, None]
    image_with_opacity = np.concatenate((image, opacity_channel), axis=-1) / 255

    unique_classes = [cls for cls in instances if len(instances[cls]) > 0]

    colours = colour_dict(category_dict, opacity)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    outlines = np.zeros(image.shape[:2])
    for cls in unique_classes:
        for instance in instances[cls]:

            mask = instance["mask"]

            if labels[cls] in categories:
                outlines[mask] = instance["id"]
                image_with_opacity[mask] = (
                    image_opacity * image_with_opacity[mask] + colours[cls]
                )

    # Outline the instances
    outlines = ski.segmentation.find_boundaries(outlines, mode="thick")
    # Equivalent to "#00ff00ff"
    image_with_opacity[outlines != 0] = np.array([0, 1, 0, 1])

    # Plot the original image and the segmented one.
    # If any of the requested categories exist in the
    # image, they will be overlaid as coloured patches
    # with the given opacity over the original image.
    # ==================================================
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[1].imshow(image_with_opacity)
    ax[1].axis("off")
    label_idx = 0

    # Add a basic class legend and extract the statistics
    for cls in unique_classes:
        if labels[cls] in categories:
            # TODO Find a way to make the label tabs the same width
            txt = ax[1].text(
                1.001,
                (1 - 2.5 * label_idx / len(labels)),
                labels[cls],
                transform=ax[1].transAxes,
                fontsize=10,
                color=1 - colours[cls][:-1],
                style="oblique",
                verticalalignment="top",
                bbox={
                    "facecolor": colours[cls][:-1] / opacity,
                    "pad": 0,
                },
            )
            label_idx += 1

    # subset = ds.sel(
    #     image=image_path.name,
    #     category=list(categories),
    # )

    stats = {}
    # for cat in categories:
    #     stats[cat] = {
    #         "colour": {
    #             stat: subset.sel(category=cat, stats=stat).colour_info.data.astype(
    #                 np.ubyte
    #             )
    #             for stat in subset.stats.data
    #         },
    #         "area": float(subset.sel(category=cat).segment.data),
    #     }

    return (fig, ax, stats)


def segment_from_dataset(
    in_file: Path | str,
    image_dir: Path | str = None,
    out_dir: Path | str = None,
    out_file: Path | str = None,
    sample: int = None,
) -> tuple[dict, Path | None]:
    """
    Segment images specified in a CSV data (sub)set.

    Args:
        input_file (tp.Union[Path, str]):
            The input dataset (a CSV file).

        image_dir (tp.Union[Path, str], optional):
            The directory where downloaded images are located. Defaults to None.

        out_dir (tp.Union[Path, str], optional):
            Directory where the output file should be saved. Defaults to None.

        out_file (tp.Union[Path, str], optional):
            Name of the output file. Defaults to None.

        sample (int, optional):
            WIP

    Returns:
        tp.Tuple[xr.Dataset, tp.Dict]:
            A tuple containing:
                - An `xarray` dataset containing the statistics
                for all the segmented images.
                - The name of the saved NetCDF file.
    """

    # Load the CSV file.
    # ==================================================
    in_file = Path(in_file).expanduser().resolve().absolute()

    try:
        df = pd.read_csv(in_file)
    except FileNotFoundError as e:
        logger.error(
            f"The provided file path '{in_file.relative_to(conf.ROOT_DIR)}' does not exist"
        )

    # Load the model
    # ==================================================
    (processor, model) = mask2former_vistas_panoptic()

    # Directory containing the downloaded images
    # ==================================================
    if image_dir is None:
        image_dir = conf.OUTPUT_DIR / "images"

    image_paths = sorted(
        [image_dir / f"{file_name}.jpeg" for file_name in df["orig_id"]]
    )
    if sample is not None:
        image_paths = image_paths[:sample]

    # Compute some image statistics
    #
    # 3D array:
    # Axis 0: image ID
    # Axis 1: category ID
    # Axis 2: statistics:
    #   - Portion of the image occupied by the category (as a fraction \in (0,1))
    #   - Colour statistics for the patch, in this order:
    #       - Median
    #       - Mode
    #       - Mean
    #       - Standard deviation
    # ==================================================
    results = []
    with tqdm(total=len(image_paths)) as pbar:
        for image_idx, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                instances = segment_image(image, processor, model)
                results.append(
                    {
                        "image": image_path.name,
                        "stats": compute_stats(image, instances),
                    }
                )
                pbar.update()

    arr = ak.Array(results)


    return (results, out_file)
