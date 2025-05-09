# --------------------------------------
from pathlib import Path

# --------------------------------------
import ibis

# --------------------------------------
from environs import Env

# --------------------------------------
from streetscapes import utils

class SVWorkspace:
    """TODO: Add docstrings"""

    @staticmethod
    def restore(path: Path):
        """
        STUB
        A method to restore a workspace from a saved session.

        Args:
            path:
                The path to the workspace root directory.
        """
        return SVWorkspace()

    def __init__(
        self,
        path: Path | str,
        env: Path | str | None = None,
        create: bool = True,
    ):
        # Directories and paths
        # ==================================================
        # The root directory of the workspace
        path = Path(path)
        if not path.exists() and not create:
            raise FileNotFoundError("The specified path does not exist.")

        self.root_dir = utils.ensure_dir(path)

        # Add a .gitignore file to the root of the workspace
        # to avoid committing the workspace accidentally
        gitignore = self.root_dir / ".gitignore"
        gitignore.touch(mode=0o750)
        with open(gitignore, "w") as gfile:
            gfile.write("/**.*\n")

        # Configuration
        # ==================================================
        self.env = Env(expand_vars=True)
        if env is None and (local_env := self.root_dir / ".env").exists():
            env = local_env

        self.env.read_env(env)

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

    def get_workspace_path(
        self,
        path: str | Path = None,
        suffix: str | None = None,
        create: bool = False,
    ):
        """
        Construct a workspace path (a file or a directory)
        with optional modifications.

        Args:
            path:
                The original path.
                Defaults to None.

            suffix:
                An optional (replacement) suffix. Defaults to None.

            create:
                Indicates that the path should be created if it doesn't exist.
                Defaults to False.

        Returns:
            The path to the file.
        """

        if path is None:
            path = self.root_dir

        path = self.root_dir / utils.make_path(
            path,
            self.root_dir,
            suffix=suffix,
        ).relative_to(self.root_dir)

        return (
            utils.ensure_dir(path) if create else path.expanduser().resolve().absolute()
        )

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

    # def add_source(
    #     self,
    #     source_type: SourceType,
    #     root_dir: str | Path | None = None,
    #     replace: bool = False,
    # ) -> SourceBase | None:
    #     """
    #     Add a source to this workspace.

    #     Args:
    #         source_type:
    #             A SourceType enum.

    #         root_dir:
    #             An optional root directory for this source. Defaults to None.

    #         replace:
    #             A switch indicating that if the source already exists,
    #             it should be replaced with the newly created one.

    #     Returns:
    #         An instance of the requested source type.
    #     """

    #     if source_type in self.sources and not replace:
    #         logger.warning(
    #             f"Reusing an existing {source_type.name} source, use the <green>replace</green> argument to override."
    #         )
    #         return self.sources[source_type]

    #     src = SourceBase.load_source(source_type, self._env, root_dir)

    #     if src is not None:
    #         self.sources[source_type] = src

    #     return src

    # def get_source(
    #     self,
    #     source_type: SourceType,
    # ) -> SourceBase | None:
    #     """
    #     Get a data source instance.

    #     Args:
    #         source_type:
    #             A SourceType enum.

    #     Returns:
    #         The data source instance, if it exists.
    #     """
    #     return self.sources.get(source_type)

    # def spawn_model(
    #     self,
    #     model_type: ModelType,
    #     replace: bool = False,
    #     *args,
    #     **kwargs,
    # ) -> ModelBase | None:
    #     """
    #     Spawn a model.

    #     Args:
    #         model_type:
    #             The model type.

    #     Returns:
    #         A model instance.
    #     """
    #     if model_type in ModelBase.models and not replace:
    #         logger.warning(
    #             f"Reusing an existing {model_type.name} model, use the <green>replace</green> argument to override."
    #         )
    #         return ModelBase.models[model_type]

    #     model = ModelBase.load_model(model_type, *args, **kwargs)

    #     if model is not None:
    #         ModelBase.models[model_type] = model

    #     return model

    # def get_model(
    #     self,
    #     model_type: ModelType,
    # ) -> ModelBase | None:
    #     """
    #     Get a model object.

    #     Args:
    #         model_type:
    #             A ModelType enum.

    #     Returns:
    #         A model instance, if it exists.
    #     """
    #     return ModelBase.models.get(model_type)

    # def save_stats(
    #     self,
    #     stats: ibis.Table,
    #     path: Path,
    # ) -> list[Path]:
    #     """
    #     Save image metadata to a Parquet file.

    #     Args:

    #         stats:
    #             A list of metadata entries.

    #         path:
    #             A directory where the stats should be saved.
    #             Defaults to None.

    #     Returns:
    #         A list of paths to the saved files.
    #     """

    #     path = self.get_workspace_path(path)
    #     path = utils.ensure_dir(path)

    #     files = []

    #     for orig_id, stat in stats.items():

    #         # File path
    #         fpath = path / f"{orig_id}.stat.parquet"

    #         # Save the stats table to a Parquet file
    #         stats.to_parquet(fpath)
    #         files.append(fpath)

    #     return files

    # def extract_stats(
    #     self,
    #     images: dict[int, np.ndarray],
    #     masks: dict[int, np.ndarray],
    #     instances: dict[int, dict],
    # ) -> dict[int, dict]:
    #     """
    #     Compute colour statistics and other metadata
    #     for a list of segmented images.

    #     Args:

    #         images:
    #             A dictionary of images as NumPy arrays.

    #         masks:
    #             A dictionary of masks as NumPy arrays.

    #         instances:
    #             A dictionary of instances containing instance-level segmentation details.
    #             Each instance is denoted with its ID (= the keys in the `instances` dictionary).

    #     Returns:
    #         dict[int, dict]:
    #             A dictionary of metadata.
    #     """

    #     logger.info("Extracting metadata...")

    #     # Statistics about instances in the images
    #     image_stats = {}

    #     # Ensure that the attrs and stats are sets
    #     attrs = set(attrs) if attrs is not None else set()
    #     stats = set(stats) if stats is not None else set()

    #     rgb = len(attrs.intersection(Attr.RGB)) > 0
    #     hsv = len(attrs.intersection(Attr.HSV)) > 0

    #     # Loop over the segmented images and compute
    #     # some statistics for each instance.
    #     for orig_id, image in images.items():

    #         # Create a new dictionary to hold the results
    #         computed = image_stats.setdefault(
    #             orig_id,
    #             {
    #                 "instance": [],
    #                 "label": [],
    #             },
    #         )

    #         # Convert the image colour space to floating-point RGB and HSV
    #         if rgb or hsv:
    #             rgb_image = ski.exposure.rescale_intensity(image, out_range=(0, 1))
    #         if hsv:
    #             hsv_image = ski.color.convert_colorspace(rgb_image, "RGB", "HSV")

    #         # Ensure that the mask and the instances for this image are available
    #         mask = masks.get(orig_id)
    #         if mask is None:
    #             continue

    #         image_instances = instances.get(orig_id)
    #         if image_instances is None:
    #             continue

    #         with tqdm(total=len(image_instances)) as pbar:
    #             for inst_id, label in image_instances.items():
    #                 computed["instance"].append(inst_id)
    #                 computed["label"].append(label)

    #                 # Extract the patches corresponding to the mask
    #                 patches = {}
    #                 inst_mask = mask == inst_id
    #                 if rgb:
    #                     rgb_patch = rgb_image[inst_mask]
    #                     patches.update(
    #                         {
    #                             Attr.R: rgb_patch[..., 0],
    #                             Attr.G: rgb_patch[..., 1],
    #                             Attr.B: rgb_patch[..., 2],
    #                         }
    #                     )
    #                 if hsv:
    #                     hsv_patch = hsv_image[inst_mask]
    #                     patches.update(
    #                         {
    #                             Attr.H: hsv_patch[..., 0],
    #                             Attr.S: hsv_patch[..., 1],
    #                             Attr.V: hsv_patch[..., 2],
    #                         }
    #                     )

    #                 # Extract the statistics for the requested attributes.
    #                 for attr in attrs:
    #                     if attr == Attr.Area:
    #                         computed.setdefault(attr, []).append(
    #                             np.count_nonzero(inst_mask)
    #                             / np.prod(rgb_image.shape[:2])
    #                         )
    #                     else:
    #                         computed.setdefault(attr, {stat: [] for stat in stats})
    #                         for stat in stats:
    #                             match stat:
    #                                 case Stat.Median:
    #                                     value = np.nan_to_num(
    #                                         np.median(patches[attr]), nan=0.0
    #                                     )
    #                                 case Stat.Mode:
    #                                     value = np.nan_to_num(
    #                                         scipy.stats.mode(patches[attr])[0], nan=0.0
    #                                     )
    #                                 case Stat.Mean:
    #                                     value = np.nan_to_num(
    #                                         np.mean(patches[attr]), nan=0.0
    #                                     )
    #                                 case Stat.SD:
    #                                     value = np.nan_to_num(
    #                                         np.std(patches[attr]), nan=0.0
    #                                     )
    #                                 case _:
    #                                     value = None

    #                             if value is not None:
    #                                 computed[attr][stat].append(value)

    #                 pbar.update()

    #     return image_stats
