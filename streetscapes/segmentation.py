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
import pandas as pd

# --------------------------------------
import xarray as xr

# --------------------------------------
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
import typing as tp

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


def segment_image(
    image: Image,
    processor: AutoImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
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
        classes = processor.post_process_semantic_segmentation(
            output, target_sizes=[image.size[::-1]]
        )[0].int()

    return classes


def mask2former_vistas_panoptic() -> (
    tp.Tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation]
):
    """
    Convenience function for loading the image processor
    and the segmentation model.

    Returns:
        tp.Tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation]:
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
    categories: tp.Dict,
    opacity: int = 255,
    cmap: str = "jet",
) -> tp.Dict:
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
    classes: pt.Tensor,
    image_idx: int,
    data: np.ndarray,
    area: np.ndarray,
):
    """
    Compute the statistics for all categories.

    Args:
        image (pt.Tensor):
            The image being processed.

        classes (pt.Tensor):
            The class mask tensor.

        image_id (int):
            The image ID.

        data (np.ndarray):
            An array holding the statistics for all the images.

        area (np.ndarray):
            An array holding the information about how much of the
            image is covered by each category.
    """

    image = np.array(image)
    unique_classes = classes.unique()
    if -1 in unique_classes:
        unique_classes = unique_classes[1:]

    for cat_id, cat in labels.items():
        mask = classes == cat_id

        # Area
        area[image_idx][cat_id] = mask.count_nonzero().item() / np.prod(image.shape[:2])

        # Statistics
        if len(mask.nonzero()) > 0:

            masked = image[mask]

            # Patch colour - median, mode, mean and SD
            data[image_idx][cat_id][0] = np.median(masked, axis=0).astype(np.ubyte)
            data[image_idx][cat_id][1] = scipy.stats.mode(masked, axis=0)[0].astype(
                np.ubyte
            )
            data[image_idx][cat_id][2] = np.mean(masked, axis=0).astype(np.ubyte)
            data[image_idx][cat_id][3] = np.std(masked, axis=0).astype(np.ubyte)


def visualise_segmentation(
    image_path: tp.Union[Path, str],
    categories: tp.List = None,
    opacity: float = 0.50,
    figsize: tp.Tuple[int, int] = (16, 8),
    ds: xr.Dataset = None,
) -> tp.Tuple[plt.Figure, plt.Axes, tp.Dict]:
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

        ds (xr.Dataset, optional):
            An `xarray` dataset for extracting the statistics. Defaults to None.

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
    classes = segment_image(image, processor, model)

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
    unique_classes = classes.unique()
    if -1 in unique_classes:
        unique_classes = unique_classes[1:]

    colours = colour_dict(category_dict, opacity)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for cls in unique_classes:
        cls = cls.item()
        idx = classes == cls
        if labels[cls] in categories:
            image_with_opacity[idx] = (
                image_opacity * image_with_opacity[idx] + colours[cls]
            )

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
        cls = cls.item()
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

    subset = ds.sel(
        image=image_path.name,
        category=list(categories),
    )

    stats = {}
    for cat in categories:
        stats[cat] = {
            "colour": {
                stat: subset.sel(category=cat, stats=stat).colour_info.data.astype(
                    np.ubyte
                )
                for stat in subset.stats.data
            },
            "area": float(subset.sel(category=cat).segment.data),
        }

    return (fig, ax, stats)


def segment_from_dataset(
    in_file: tp.Union[Path, str],
    image_dir: tp.Union[Path, str] = None,
    out_dir: tp.Union[Path, str] = None,
    out_file: tp.Union[Path, str] = None,
    sample: int = None,
) -> tp.Tuple[xr.Dataset, tp.Dict]:
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
    stat_names = ["median", "mode", "mean", "sd"]
    data = np.zeros((len(image_paths), len(labels), len(stat_names), 3))
    area = np.zeros((len(image_paths), len(labels)))
    with tqdm(total=len(image_paths)) as pbar:
        for image_idx, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                classes = segment_image(image, processor, model)
                compute_stats(image, classes, image_idx, data, area)
                pbar.update()

    ds = xr.Dataset(
        {
            "colour_info": (["image", "category", "stats", "colour"], data),
            "segment": (["image", "category"], area),
        },
        coords={
            "image": [path.name for path in image_paths],
            "category": list(labels.values()),
            "stats": stat_names,
            "colour": ["R", "G", "B"],
        },
    )

    # Save to a NetCDF file
    # ==================================================
    # Directory for the output file(s)
    if out_dir is None:
        out_dir = conf.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the output file name
    if out_file is None:
        out_file = f"{in_file.stem}-stats.nc"
    out_file = out_dir / out_file

    # Save the file
    ds.to_netcdf(out_file)

    return (ds, out_file)
