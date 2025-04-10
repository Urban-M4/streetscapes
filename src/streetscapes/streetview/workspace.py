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
from environs import Env

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
        conf: Path | str | None = None,
        create: bool = False,
    ):
        # Directories and paths
        # ==================================================
        # The root directory of the workspace
        self.root_dir = Path(path)
        if not self.root_dir.exists():
            if not create:
                raise FileNotFoundError("The specified path does not exist.")
            utils.ensure_dir(path)

        # Configuration
        # ==================================================
        self._env = Env(expand_vars=True)
        if conf is None and (local_env := Path.cwd() / ".env").exists():
            conf = local_env

        self._env.read_env(conf)

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

    def get_path(
        self,
        path: str | Path,
        suffix: str | None = None,
    ):
        """
        Construct a workspace path (a file or a directory)
        with optional modifications.

        Args:
            path:
                The original path.

            suffix:
                An optional (replacement) suffix. Defaults to None.

        Returns:
            The path to the file.
        """

        return utils.make_path(path, self.root_dir, suffix=suffix)

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

        filename = self.get_path(filename, suffix="csv")

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

        filename = self.get_path(filename, suffix="parquet")

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
                f"Reusing an existing {source_type.name}, use the <green>replace</green> argument to override."
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
            if src not in self.sources:
                continue

            source = self.sources[src]
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
    ) -> list[Path]:
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
            A list of image paths.
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
        fpath = self.get_path(dataset, suffix="parquet")

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

                dataset = self.workspace.load_parquet(fpath)
                if columns is not None:
                    dataset = dataset.select(columns)

                pbar.update()

            pbar.set_description_str(f"{desc} | Done")

        return (dataset, fpath)

    def _save_results(
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

    def save_stats(
        self,
        stats: dict,
        path: Path | None = None,
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
            list[Path]:
                A list of paths to the saved files.
        """

        if path is None:
            path = conf.PARQUET_DIR

        path = scs.mkdir(path)

        files = []

        for orig_id, stat in stats.items():

            # File path
            fpath = path / f"{orig_id}.stat.parquet"

            # Convert the stats into a JSON object and
            # then into an Awkward array.
            arr = ak.from_json(json.dumps(stat))

            # Save the array to a Parquet file.
            ak.to_parquet(arr, fpath)

            files.append(fpath)

        return files

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
                self._save_results(results, source.root_dir, model_type)
                segmentations.extend(results)
                pbar.update()

            pbar.set_description_str("Done")

        segmentations = [
            SVSegmentation(model_type, segmentation["image_path"])
            for segmentation in segmentations
        ]

        return segmentations
