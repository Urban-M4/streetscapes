# --------------------------------------
import os

# --------------------------------------
from pathlib import Path

# --------------------------------------
import itertools

# --------------------------------------
import ibis

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import numpy as np

# --------------------------------------
import skimage as ski

# --------------------------------------
from environs import Env

# --------------------------------------
from typing import Any

# --------------------------------------
from streetscapes import utils
from streetscapes.utils import logger
from streetscapes.models import ModelBase
from streetscapes.models import ModelType
from streetscapes.sources import SourceBase
from streetscapes.sources import SourceType
from streetscapes.sources import ImageSourceBase
from streetscapes.streetview import SVSegmentation


class SVWorkspace:

    @staticmethod
    def restore(path: Path):
        """
        STUB
        A method to restore a workspace from a saved session.

        Args:
            path:
                The path to the workspace root directory.
        """
        pass

    def __init__(
        self,
        path: Path | str,
        env: Path | str | None = None,
        create: bool = False,
    ):
        # Directories and paths
        # ==================================================
        # The root directory of the workspace
        path = Path(path)
        if not path.exists():
            if not create:
                raise FileNotFoundError("The specified path does not exist.")
            path = utils.ensure_dir(path)

        self.root_dir = path.expanduser().resolve().absolute()

        # Configuration
        # ==================================================
        self._env = Env(expand_vars=True)
        if env is None and (local_env := self.root_dir / ".env").exists():
            env = local_env

        self._env.read_env(env)

        # Sources
        # ==================================================
        self.sources = {}

        # Some internal convenience attributes
        # ==================================================
        self._source_col = "source"
        self._id_col = "image_id"

        # Metadata object.
        # Can be used to save and reload a workspace.
        # ==================================================
        self.metadata = self._load_metadata()

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(root_dir={utils.hide_home(self.root_dir)!r})"

    def _load_metadata(self) -> ibis.Table:

        metadata = self.root_dir / "metadata.db"
        if not metadata.exists():
            ibis.connect(f"duckdb://{metadata}")

    def _save_segmentations(
        self,
        segmentations: list[dict],
        path: Path,
        model_type: ModelType,
    ):
        """
        Save image segmentation masks to NPZ files.

        Args:

            segmentations:
                A list of dictionaries containing image segmentation information.

            path:
                The path to use for saving segmentations.

            model_type:
                The model used for segmenting the images.
        """

        # Make sure that the paths exists
        path = utils.ensure_dir(path)

        # The model name
        # Masks and instances are saved in separate directories for each model.
        model_name = model_type.name.lower()
        mask_dir = utils.ensure_dir(utils.make_path(f"masks/{model_name}", path))
        instance_dir = utils.ensure_dir(
            utils.make_path(f"instances/{model_name}", path)
        )

        for segmentation in segmentations:

            # Check if this mask has already been saved.
            if "mask_path" in segmentation:
                continue

            image_path = segmentation["image_path"]

            # Save the mask as a compressed NumPy array.
            # ==================================================
            mask_path = mask_dir / image_path.with_suffix(".npz").name
            np.savez_compressed(mask_path, segmentation["mask"])
            segmentation["mask"] = mask_path

            # Save the instances as a Parquet file
            # ==================================================
            instance_path = instance_dir / image_path.with_suffix(".parquet").name

            ibis.memtable(
                list(segmentation["instances"].items()),
                columns=["instance", "label"],
            ).to_parquet(instance_path)
            segmentation["instances"] = instance_path

    def get_workspace_path(
        self,
        path: str | Path,
        suffix: str | None = None,
        create: bool = False,
    ):
        """
        Construct a workspace path (a file or a directory)
        with optional modifications.

        Args:
            path:
                The original path.

            suffix:
                An optional (replacement) suffix. Defaults to None.

            create:
                Indicates that the path should be created if it doesn't exist.
                Defaults to False.

        Returns:
            The path to the file.
        """

        path = self.root_dir / utils.make_path(
            path,
            self.root_dir,
            suffix=suffix,
        ).relative_to(self.root_dir)

        return utils.ensure_dir(path) if create else path.expanduser().resolve().absolute()

    def load_csv(
        self,
        filename: str | Path,
    ) -> ibis.Table:
        """
        Load a CSV file from the current workspace.

        Args:
            filename:
                The path to the file.

        Returns:
            An Ibis table.
        """

        filename = self.get_workspace_path(filename, suffix="csv")

        return ibis.read_csv(filename)

    def load_parquet(
        self,
        filename: str | Path,
    ) -> ibis.Table:
        """
        Load a Parquet file from the current workspace.

        Args:
            filename:
                The path to the file.

        Returns:
            An Ibis table.
        """

        filename = self.get_workspace_path(filename, suffix="parquet")

        return ibis.read_parquet(filename)

    def show_contents(self) -> str | None:
        """
        Create and return a tree-like representation of a directory.
        """
        return utils.show_dir_tree(self.root_dir)

    def add_source(
        self,
        source_type: SourceType,
        root_dir: str | Path | None = None,
        replace: bool = False,
    ) -> SourceBase | None:
        """
        Add a source to this workspace.

        Args:
            source_type:
                A SourceType enum.

            root_dir:
                An optional root directory for this source. Defaults to None.

            replace:
                A switch indicating that if the source already exists,
                it should be replaced with the newly created one.

        Returns:
            An instance of the requested source type.
        """

        if source_type in self.sources and not replace:
            logger.warning(
                f"Reusing an existing {source_type.name} source, use the <green>replace</green> argument to override."
            )
            return self.sources[source_type]

        src = SourceBase.load_source(source_type, self._env, root_dir)

        if src is not None:
            self.sources[source_type] = src

        return src

    def get_source(
        self,
        source_type: SourceType,
    ) -> SourceBase | None:
        """
        Get a data source instance.

        Args:
            source_type:
                A SourceType enum.

        Returns:
            The data source instance, if it exists.
        """
        return self.sources.get(source_type)

    def spawn_model(
        self,
        model_type: ModelType,
        replace: bool = False,
        *args,
        **kwargs,
    ) -> ModelBase | None:
        """
        Spawn a model.

        Args:
            model_type:
                The model type.

        Returns:
            A model instance.
        """
        if model_type in ModelBase.models and not replace:
            logger.warning(
                f"Reusing an existing {model_type.name} model, use the <green>replace</green> argument to override."
            )
            return ModelBase.models[model_type]

        model = ModelBase.load_model(model_type, *args, **kwargs)

        if model is not None:
            ModelBase.models[model_type] = model

        return model

    def get_model(
        self,
        model_type: ModelType,
    ) -> ModelBase | None:
        """
        Get a model object.

        Args:
            model_type:
                A ModelType enum.

        Returns:
            A model instance, if it exists.
        """
        return ModelBase.models.get(model_type)

    def get_source_types_from_table(
        self,
        table: ibis.Table,
    ) -> set:
        """
        Return a set of SourceType enums created by filtering the
        table and checking what sources are contained inside.

        Args:
            table: An Ibis table.

        Returns:
            A set of SourceType enums that can be used to perform custom actions by source type.
        """

        sources = set()
        for s in table.select(self._source_col).distinct().to_pandas().values:
            try:
                sources.add(SourceType(s[0]))

            except:
                continue

        return sources

    def check_image_status(
        self,
        dataset: ibis.Table,
    ) -> tuple[set, set]:
        """
        Get the IDs of images that are missing from the local root directory.

        This method expects the colums corresponding to the source and the image ID
        to be named in a certain way (cf. self._source_col and self._id_col, respectively).
        This can be easily handled with Ibis by using .select() with a dictionary argument.
        For instance, assuming a table that contains columns named "source" and "orig_id"
        (as in the case of the Global Streetscapes dataset), we can obtain a new table
        with columns named "source" and "image_id" by passing a dictionary mapping the
        new column names to the existing ones:

        >>> t.select("source", "orig_id").columns
        ('source', 'orig_id')

        >>> t.select({'source': "source", "image_id": "orig_id"}).columns
        ('source', 'image_id')

        Here, 'source' is mapped unchanged to the original column called 'source'.

        Args:
            dataset:
                A dataset containing information about images that can be downloaded.

        Returns:
            A tuple containing:
                1. A set of existing images.
                2. A set of missing images.
        """

        sources = self.get_source_types_from_table(dataset)

        existing = {}
        missing = {}

        for src in sources:

            # Get the source object or add it if it's missing
            source = self.sources.get(src, self.add_source(src))
            if source is None:
                continue

            if isinstance(source, ImageSourceBase):

                filtered = [
                    str(s)
                    for s in dataset.filter(
                        dataset[self._source_col].ilike(f"%{src.name}")
                    )
                    .select(self._id_col)
                    .to_pandas()
                    .to_numpy()[:, 0]
                    .tolist()
                ]

                _existing, _missing = source.check_image_status(filtered)

                existing[src] = _existing
                missing[src] = _missing

        return existing, missing

    def download_images(
        self,
        dataset: ibis.Table,
        sample: int | None = None,
        max_workers: int = None,
    ) -> ibis.Table:
        """
        Download a set of images concurrently.

        Args:
            dataset:
                A dataset containing image IDs.

            sample:
                Only download a sample of the images. Defaults to None.

            max_workers:
                The number of workers (threads) to use. Defaults to None.

        Returns:
            An Ibis table containing information about the downloaded images.
        """
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import as_completed

        # Limit the table if only a sample is required
        total = dataset.count().as_scalar().to_pandas()
        if isinstance(sample, int) and sample < total:
            dataset = ibis.memtable(dataset.to_pandas().sample(sample)).as_table()

        # Get the IDs of images that haven't been downloaded yet.
        (existing, missing) = self.check_image_status(dataset)

        # Download missing images
        for source_type, image_ids in missing.items():

            # Download the images in parallel
            # ==================================================
            if len(image_ids) > 0:

                if max_workers is None:
                    max_workers = os.cpu_count()

                source = self.sources[source_type]

                with ThreadPoolExecutor(max_workers=max_workers) as tpe:

                    # Submit the image IDs for processing
                    futures = {
                        tpe.submit(
                            source.download_image,
                            image_id,
                        ): image_id
                        for image_id in image_ids
                    }

                    desc = f"Downloading images | {source_type.name}"
                    with tqdm(total=len(image_ids), desc=desc) as pbar:
                        for future in as_completed(futures):
                            try:
                                image_id = futures[future]
                                result = future.result()
                                existing[source_type].add(result)
                                pbar.set_description_str(f"{desc} | {image_id:>20s}")
                            except Exception as exc:
                                logger.info(
                                    f"Error downloading image {image_id}:\n{exc}"
                                )
                            pbar.update()
                        pbar.set_description_str(f"{desc} | Done")

        return dataset

    def load_dataset(
        self,
        source: SourceType | SourceBase,
        dataset: str,
        criteria: dict = None,
        columns: list | tuple | set = None,
        recreate: bool = False,
        save: bool = True,
    ) -> ibis.Table | None:
        """
        Load and return a subset of the source, if it exists.

        Args:
            source:
                The source to use.

            dataset:
                The dataset to load.

            criteria:
                Optional criteria used to create a subset.

            columns:
                The columns to keep or retrieve.

            recreate:
                Recreate the dataset if it exists.
                Defaults to False.

            save:
                Save a newly created dataset.
                Defaults to True.

        Returns:
            An Ibis table.
        """

        # The path to the dataset.
        fpath = self.get_workspace_path(dataset, suffix="parquet")

        desc = f"Dataset {dataset}"
        with tqdm(total=1, desc=desc) as pbar:
            if recreate or not fpath.exists():

                # Get the actual source if only the type is provided
                if isinstance(source, SourceType):
                    source = self.sources.get(source)
                if source is None:
                    return

                pbar.set_description_str(f"{desc} | Extracting...")

                dataset = source.load_dataset(criteria, columns)

                if save:
                    pbar.set_description_str(f"{desc} | Saving...")
                    utils.ensure_dir(fpath.parent)
                    dataset.to_parquet(fpath)

                pbar.update()
            else:
                pbar.set_description_str(f"{desc} | Loading...")

                dataset = self.load_parquet(fpath)
                if columns is not None:
                    dataset = dataset.select(columns)

                pbar.update()

            pbar.set_description_str(f"{desc} | Done")

        return (dataset, fpath)

    def save_stats(
        self,
        stats: ibis.Table,
        path: Path,
    ) -> list[Path]:
        """
        Save image metadata to a Parquet file.

        Args:

            stats:
                A list of metadata entries.

            path:
                A directory where the stats should be saved.
                Defaults to None.

        Returns:
            A list of paths to the saved files.
        """

        path = self.get_workspace_path(path)
        path = utils.ensure_dir(path)

        files = []

        for orig_id, stat in stats.items():

            # File path
            fpath = path / f"{orig_id}.stat.parquet"

            # Save the stats table to a Parquet file
            stats.to_parquet(fpath)
            files.append(fpath)

        return files

    def extract_stats(
        self,
        images: dict[int, np.ndarray],
        masks: dict[int, np.ndarray],
        instances: dict[int, dict],
    ) -> dict[int, dict]:
        """
        Compute colour statistics and other metadata
        for a list of segmented images.

        Args:

            images:
                A dictionary of images as NumPy arrays.

            masks:
                A dictionary of masks as NumPy arrays.

            instances:
                A dictionary of instances containing instance-level segmentation details.
                Each instance is denoted with its ID (= the keys in the `instances` dictionary).

        Returns:
            dict[int, dict]:
                A dictionary of metadata.
        """

        logger.info("Extracting metadata...")

        # Statistics about instances in the images
        image_stats = {}

        # Ensure that the attrs and stats are sets
        attrs = set(attrs) if attrs is not None else set()
        stats = set(stats) if stats is not None else set()

        rgb = len(attrs.intersection(Attr.RGB)) > 0
        hsv = len(attrs.intersection(Attr.HSV)) > 0

        # Loop over the segmented images and compute
        # some statistics for each instance.
        for orig_id, image in images.items():

            # Create a new dictionary to hold the results
            computed = image_stats.setdefault(
                orig_id,
                {
                    "instance": [],
                    "label": [],
                },
            )

            # Convert the image colour space to floating-point RGB and HSV
            if rgb or hsv:
                rgb_image = ski.exposure.rescale_intensity(image, out_range=(0, 1))
            if hsv:
                hsv_image = ski.color.convert_colorspace(rgb_image, "RGB", "HSV")

            # Ensure that the mask and the instances for this image are available
            mask = masks.get(orig_id)
            if mask is None:
                continue

            image_instances = instances.get(orig_id)
            if image_instances is None:
                continue

            with tqdm(total=len(image_instances)) as pbar:
                for inst_id, label in image_instances.items():
                    computed["instance"].append(inst_id)
                    computed["label"].append(label)

                    # Extract the patches corresponding to the mask
                    patches = {}
                    inst_mask = mask == inst_id
                    if rgb:
                        rgb_patch = rgb_image[inst_mask]
                        patches.update(
                            {
                                Attr.R: rgb_patch[..., 0],
                                Attr.G: rgb_patch[..., 1],
                                Attr.B: rgb_patch[..., 2],
                            }
                        )
                    if hsv:
                        hsv_patch = hsv_image[inst_mask]
                        patches.update(
                            {
                                Attr.H: hsv_patch[..., 0],
                                Attr.S: hsv_patch[..., 1],
                                Attr.V: hsv_patch[..., 2],
                            }
                        )

                    # Extract the statistics for the requested attributes.
                    for attr in attrs:
                        if attr == Attr.Area:
                            computed.setdefault(attr, []).append(
                                np.count_nonzero(inst_mask)
                                / np.prod(rgb_image.shape[:2])
                            )
                        else:
                            computed.setdefault(attr, {stat: [] for stat in stats})
                            for stat in stats:
                                match stat:
                                    case Stat.Median:
                                        value = np.nan_to_num(
                                            np.median(patches[attr]), nan=0.0
                                        )
                                    case Stat.Mode:
                                        value = np.nan_to_num(
                                            scipy.stats.mode(patches[attr])[0], nan=0.0
                                        )
                                    case Stat.Mean:
                                        value = np.nan_to_num(
                                            np.mean(patches[attr]), nan=0.0
                                        )
                                    case Stat.SD:
                                        value = np.nan_to_num(
                                            np.std(patches[attr]), nan=0.0
                                        )
                                    case _:
                                        value = None

                                if value is not None:
                                    computed[attr][stat].append(value)

                    pbar.update()

        return image_stats

    def segment_from_dataset(
        self,
        dataset: ibis.Table,
        model: ModelType | ModelBase,
        labels: dict,
        ensure: list[str] | None = None,
        batch_size: int = 10,
        download: bool = True,
    ) -> ibis.Table:
        """
        Retrieve the paths of local images from a dataset.

        Args:
            dataset:
                The dataset (an Ibis table).

            model:
                A ModelType or a model instance to use for the segmentation.

            labels:
                A flattened set of labels to look for,
                with optional subsets of labels that should be
                checked in order to eliminate overlaps.
                Cf. `BaseSegmenter._flatten_labels()`

            ensure:
                A list of columns whose values must be set (not null or empty).
                Defaults to None.

            batch_size:
                Process the images in batches of this size.
                Defaults to 10.

            download:
                A toggle indicating whether missing images should be downloaded.
                Defaults to True.

        Returns:
            A table of information about the segmentations.
        """

        if ensure is not None:
            dataset = dataset.drop_null(ensure)

        existing, missing = self.check_image_status(dataset)

        if download:
            # TODO: Download the missing images.
            # For now, we rely on the user having done that already.
            pass

        if isinstance(model, ModelType):
            model = self.get_model(model)

        if model is None:
            logger.warning(f"Error loading model of type {model_type}")
            return

        # Segment the images and save the results
        # ==================================================
        segmentations = []

        for source_type, image_paths in existing.items():

            source = self.get_source(source_type)
            if source is None or not isinstance(source, ImageSourceBase):
                logger.warning(
                    f"Invalid image source '{source_type.name}', moving on..."
                )
                continue

            # Compute the number of batches
            total = len(image_paths) // batch_size
            if total * batch_size != len(image_paths):
                total += 1

            # Model type (used for saving the metadata)
            model_type = model.get_model_type()

            # Segment the images and extract the metadata
            pbar = tqdm(total=total, desc=f"Segmenting {source_type.name} images...")
            for path_batch in itertools.batched(list(image_paths), batch_size):
                results = model.segment_images(path_batch, labels)
                self._save_segmentations(results, source.root_dir, model_type)
                segmentations.extend(results)
                pbar.update()

            pbar.set_description_str("Done")

        segmentations = [
            SVSegmentation(model_type, segmentation["image_path"])
            for segmentation in segmentations
        ]

        return segmentations
