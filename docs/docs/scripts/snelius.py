# --------------------------------------
import torch

# --------------------------------------
import ibis

ibis.options.interactive = True

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
from streetscapes import logger
from streetscapes import utils
from streetscapes.models import ModelType
from streetscapes.sources import SourceType
from streetscapes.streetview import SVWorkspace


def main():

    ws = SVWorkspace("./Amsterdam", create=True)
    gss = ws.add_source(SourceType.GlobalStreetscapes)
    mp = ws.add_source(SourceType.Mapillary)
    kv = ws.add_source(SourceType.KartaView)

    logger.info(f"Global Streetscapes source:\n{gss}")

    # Subset name (path relative to the root directory of the workspace + file name without the .parquet extension)
    subset = "subsets/amsterdam"

    # Criteria used to filter the large Global Streetscapes dataset.
    criteria = {"city": "Amsterdam", "view_direction": "side"}

    # Columns to keep as in the subset.
    columns = {"uuid", "source", "city", "lat", "lon", "orig_id", "source"}

    # Create the subset and assign it to a variable that we can use below.
    # The method also returns the path to the saved subset if the dataset was saved to disk (triggered by save=True).
    (ams, ams_path) = ws.load_dataset(
        gss,
        subset,
        criteria=criteria,
        columns=columns,
    )

    # Show how many entries we have
    logger.info(f"{subset} ]Rows:\n{ams.count()}")

    # Source table - remap columns to match the required interface
    src_table = ams.select(
        {
            "source": "source",
            "image_id": "orig_id",
        }
    )

    # Download a sample of the images
    sample = ws.download_images(src_table, sample=10)

    logger.info(f"Sample size:\n{sample.count()}")

    existing, missing = ws.check_image_status(sample)
    logger.info(
        f"Image count ] Downloaded: {sum([len(v) for v in existing.values()])} | missing: {sum([len(v) for v in missing.values()])}"
    )

    try:
        # Spawn the DinoSAM model
        torch.cuda.empty_cache()
        ds = ws.spawn_model(ModelType.DinoSAM)

        # Labels
        labels = {
            "building": {
                "window": None,
                "door": None,
            },
            "vegetation": None,
            "car": None,
            "road": None,
            "bicycle": None,
            "bicycle": None,
            "water": None,
            # Some materials that are not associated with specific objects
            "brick": None,
            "asphalt": None,
            "concrete": None,
            "glass": None,
            "metal": None,
            "rubber": None,
            "wood": None,
            "plastic": None,
        }

        # Segment the images associated with our subset
        ds_segmentations = ws.segment_from_dataset(sample, ds, labels, batch_size=10)

        plot_dir = ws.get_workspace_path("plots", create=True)

        for seg in tqdm(ds_segmentations):

            # Segment the images associated with our subset
            labels = seg.get_instance_table().select("label").distinct()

            # Create a visualisation
            fig, axes = seg.visualise()

            # Save the visualisation
            fig.savefig(plot_dir / seg.image_path.name)

    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
