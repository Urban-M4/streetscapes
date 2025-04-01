# --------------------------------------
from __future__ import annotations

# --------------------------------------
import os

# --------------------------------------
import json

# --------------------------------------
import time

# --------------------------------------
import random

# --------------------------------------
from pathlib import Path

# --------------------------------------
import operator

# --------------------------------------
import requests

# --------------------------------------
from functools import reduce

# --------------------------------------
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# --------------------------------------
import sys

# --------------------------------------
import ibis

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import numpy as np

# --------------------------------------
import pandas as pd

# --------------------------------------
import platform

# --------------------------------------
import skimage as ski

# --------------------------------------
from environs import Env

# --------------------------------------
import requests as rq

# --------------------------------------
from huggingface_hub import hf_hub_download
from huggingface_hub import CachedRepoInfo
from huggingface_hub import try_to_load_from_cache
from huggingface_hub import cached_assets_path
from huggingface_hub import scan_cache_dir


# --------------------------------------
import matplotlib.pyplot as plt

# --------------------------------------
import typing as tp

# --------------------------------------
from streetscapes.utils import enums
from streetscapes.utils.enums import Source
from streetscapes.utils.logging import logger


class SVWorkspace:

    @staticmethod
    def ensure_dir(path: Path | str) -> Path:
        """
        Resolve and expand a directory path and
        create the directory if it doesn't exist.

        Args:
            directory:
                A directory path.

        Returns:
            The (potentially newly created) expanded path.
        """

        path = Path(path).expanduser().resolve().absolute()
        path.mkdir(exist_ok=True, parents=True)
        return path

    @staticmethod
    def create(
        path: Path | str,
        conf: Path | str | None = None,
    ) -> SVWorkspace:
        """
        Create a workspace.

        Args:
            path:
                The path to the workspace directory.

            conf:
                A configuration file. Defaults to None.

        Raises:
            FileExistsError:
                Raised if the directory already exists.

        Returns:
            The created workspace.
        """

        path = Path(path)
        if path.exists():
            raise FileExistsError("The specified path already exists.")

        # Return a workspace
        return SVWorkspace(SVWorkspace.ensure_dir(path), conf)

    def __init__(
        self,
        path: Path | str,
        conf: Path | str | None = None,
    ):

        # Configuration
        # ==================================================
        self._env = Env(expand_vars=True)
        if conf is None and (local_env := Path.cwd() / ".env").exists():
            conf = local_env

        self._env.read_env(conf)

        # Repository details
        # ==================================================
        # self._repo_id = "NUS-UAL/global-streetscapes"
        # self._repo_type = "dataset"

        # Directories and paths
        # ==================================================
        # The root directory of the workspace
        self.root_dir = Path(path)
        if not self.root_dir.exists():
            raise FileNotFoundError("The specified path does not exist.")

        self.sources = set()


    # def _bootstrap(self):

    #     # Optional local directory for downloading data from the Global Streetscapes repo.
    #     # Defaults to the local Huggingface cache directory.
    #     self.local_dir = self._env.path("STREETSCAPES_LOCAL_DIR", None)

    #     # Load the info file.
    #     self.info = self.get_global_file_path("info.csv")

    #     # Now scan the cache directory to extract the paths
    #     cache_dir = scan_cache_dir()
    #     for repo in cache_dir.repos:
    #         if repo.repo_id == self._repo_id:
    #             self.local_dir = repo.repo_path

    #     # Bootstrap subdirectories
    #     # ==================================================
    #     self.csv_dir = self.local_dir / "data"
    #     self.parquet_dir = self.csv_dir / "parquet"

    #     # Bootstrap image subdirectories
    #     # ==================================================
    #     self.image_dirs = {
    #         src: self.create_image_dir(src.name.lower()) for src in Source
    #     }

    def construct_path(
        self,
        path: str | Path,
        parent: Path | None = None,
        root: Path | None = None,
        suffix: str | None = None,
    ):
        """
        Construct a path (a file or a directory) with optional modifications.

        Args:
            path:
                The original path.

            parent:
                A parent path. Defaults to None.

            root:
                An optional root path for computing a relative path.
                Defaults to None.

            suffix:
                An optional (replacement) suffix. Defaults to None.

        Returns:
            The path to the file.
        """

        # Ensure that the path is a Path object.
        path = Path(path)

        # Optionally position the path relative to a parent path.
        if not path.is_absolute():
            if parent is None:
                parent = self.root_dir

            if root is None:
                root = self.root_dir

            path = parent.relative_to(root) / path

        # Optionally replace or add a suffix.
        if suffix is not None:
            path = path.with_suffix(f".{suffix}")

        return path

    def load_csv_file(
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

        filename = self.construct_path(filename, suffix="csv")

        return ibis.read_csv(filename)

    def load_parquet_file(
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

        filename = self.construct_path(filename, suffix="parquet")

        return ibis.read_parquet(filename)

    def create_image_dir(
        self,
        subdir: str,
    ) -> Path:
        """
        Create a managed image directory for a given image source.

        Args:
            subdir:
                A subdirectory corresponding to the image source.

        Returns:
            A Path to the image subdirectory.
        """

        return Path(
            cached_assets_path(
                library_name="streetscapes",
                namespace="images",
                subfolder=subdir,
            )
        )

    # def get_missing_image_ids(
    #     records: pd.DataFrame,
    #     directory: Path,
    #     image_paths: set[Path] = None,
    # ) -> tuple[set[Path], set[int]]:
    #     """
    #     Extract the set of IDs for images that have not been downloaded yet.

    #     Args:
    #         records (pd.DataFrame):
    #             A Pandas dataframe containing image ID information.

    #         directory (Path):
    #             The directory to search.

    #         image_paths (set[Path], optional):
    #             A set of paths that has already been collected. Defaults to None.

    #     Returns:
    #         tuple[set[Path], set[int]]:
    #             A tuple containing:
    #                 1. A set of paths to existing images.
    #                 2. A set of image IDs to download.
    #     """

    #     if image_paths is None:
    #         image_paths = set()
    #     missing = set()
    #     for record in records:
    #         image_id = record["orig_id"]
    #         image_path = directory / f"{image_id}.jpeg"
    #         if image_path.exists():
    #             image_paths.add(image_path)
    #         else:
    #             missing.add(image_id)

    #     return (image_paths, missing)

    # def get_image_url(
    #     image_id: int,
    #     source: enums.Source,
    #     resolution: int = 2048,
    #     session: rq.Session = None,
    # ) -> str:
    #     """
    #     Retrieve the URL for an image with the given ID.

    #     Args:
    #         source (enums.Source):
    #             The source map (cf. the SourceMap enum).

    #         image_id (int):
    #             The image ID.

    #         resolution (int):
    #             The resolution of the requested image (only valid for Mapillary for now).

    #         session (rq.Session, optional):
    #             An optional authenticated session to use
    #             for retrieving the image.

    #     Returns:
    #         str:
    #             The URL to query.
    #     """

    #     match source:
    #         case enums.Source.Mapillary:
    #             url = f"https://graph.mapillary.com/{image_id}?fields=thumb_{resolution}_url"

    #             try:
    #                 prq = rq.Request(
    #                     "GET", url, params={"access_token": conf.MAPILLARY_TOKEN}
    #                 )
    #                 res = session.send(prq.prepare())
    #                 image_url = json.loads(res.content.decode("utf-8"))[
    #                     f"thumb_{resolution}_url"
    #                 ]

    #                 return image_url

    #             except Exception as e:
    #                 return

    #         case enums.Source.KartaView:
    #             url = f"https://api.openstreetcam.org/2.0/photo/?id={image_id}"
    #             try:
    #                 # Send the request
    #                 r = rq.get(url)
    #                 # Parse the response
    #                 data = r.json()["result"]["data"][0]
    #                 image_url = data["fileurlProc"]
    #                 return image_url
    #             except Exception as e:
    #                 return

    #         case _:
    #             return

    # def download_image(
    #     image_id: int,
    #     directory: str | Path,
    #     source: enums.Source,
    #     resolution: int = 2048,
    #     verbose: bool = True,
    #     session: rq.Session = None,
    # ) -> Path:
    #     """
    #     Download a single image from Mapillary.

    #     Args:
    #         image_id (str):
    #             The image ID.

    #         directory (str | Path):
    #             The destination directory.

    #         source (enums.SourceMap):
    #             The source map.
    #             Limited to Mapillary or KartaView at the moment.

    #         resolution (int, optional):
    #             The resolution to request. Defaults to 2048.

    #         verbose (bool, optional):
    #             Print some output. Defaults to True.

    #         session (rq.Session, optional):
    #             An optional authenticated session to use
    #             for retrieving the image.

    #     Returns:
    #         Path:
    #             The path to the downloaded image file.
    #     """

    #     # Set up the image path
    #     image_path = directory / f"{image_id}.jpeg"

    #     # Download the image
    #     if not image_path.exists():
    #         if verbose:
    #             logger.info(f"Downloading image {image_id}.jpeg...")

    #         # Random sleep time so that we don't flood the servers.
    #         time.sleep(random.uniform(0.1, 1))

    #         # NOTE: Specifically in the case of Mapillary,
    #         # we have to send a request for that image
    #         # straight after getting the URL.
    #         # Collecting all the URLs in advance and requesting them
    #         # one by one outside the loop doesn't work.

    #         # Download the image
    #         # ==================================================
    #         match source:
    #             case enums.Source.Mapillary:
    #                 if session is None:
    #                     session = get_session(source)
    #                 url = get_image_url(image_id, source, resolution, session)
    #                 response = session.get(url)
    #             case enums.Source.KartaView:
    #                 url = get_image_url(image_id, source, resolution)
    #                 response = rq.get(url)

    #         # Save the image if it has been downloaded successfully
    #         # ==================================================
    #         if response.status_code == 200 and response.content is not None:
    #             with open(image_path, "wb") as f:
    #                 f.write(response.content)

    #     return image_path


    def download_images(
        df: ibis.Table,
        sample: bool = False,
        max_workers: int = None,
    ) -> list[Path]:
        """
        Download a set of images concurrently.

        Args:
            df:
                A dataframe containing image IDs.

            sample:
                Only download a sample set of images. Defaults to None.

            max_workers:
                The number of workers (threads) to use. Defaults to None.

            verbose:
                Print some output. Defaults to False.

        Returns:
            A list of image paths.
        """

        # Filter records by source
        filtered = {}
        for source in enums.Source:
            subset = df[df["source"].str.lower() == source.name.lower()]
            if len(subset) > 0:
                filtered[source] = subset

        # Set up the image directory
        directory = ensure_dir(directory)

        image_paths = set()
        for source, records in filtered.items():
            # Limit the records if only a sample is required
            if isinstance(sample, int):
                records = records.sample(sample)

            # Convert records to a dictionary
            records = records.to_dict("records")

            # Get the IDs of images that haven't been downloaded yet.
            (image_paths, missing) = get_missing_image_ids(
                records, directory, image_paths
            )

            # Download the images in parallel
            # ==================================================
            if len(missing) > 0:
                # Authenticated session for this source (if necessary)
                session = get_session(source)

                if max_workers is None:
                    max_workers = os.cpu_count()
                with ThreadPoolExecutor(max_workers=max_workers) as tpe:
                    logger.info(
                        f"Downloading {len(missing)} images from {source.name} into '{directory.name}'..."
                    )

                    # Submit the image IDs for processing
                    futures = {
                        tpe.submit(
                            download_image,
                            image_id,
                            directory,
                            source,
                            resolution,
                            verbose,
                            session,
                        ): image_id
                        for image_id in missing
                    }

                    with tqdm(total=len(missing), desc="Download progress: ") as pbar:
                        for future in as_completed(futures):
                            try:
                                image_id = futures[future]
                                image_paths.add(future.result())
                                if verbose:
                                    logger.info(
                                        f"Image {image_id} downloaded successuflly"
                                    )
                            except Exception as exc:
                                logger.debug(f"Error downloading image {image_id}")
                            pbar.update(1)

        return image_paths

    def _add_source(self, source: Source):
        self.sources.append(source)

    def as_rgb(
        image: np.ndarray,
        greyscale: bool = False,
    ) -> np.ndarray:
        """
        Convert an image into an RGB version.

        Args:
            image (np.ndarray):
                The image to convert.

            greyscale (bool, optional):
                Switch to convert the image to greyscale.
                Defaults to False.

        Returns:
            np.ndarray:
                The RGB image.
        """

        if len(image.shape) == 2:
            # The image is already greyscale.
            # Just convert it to RGB.
            image = ski.color.gray2rgb(image)

        else:
            if image.shape[-1] == 4:
                # Remove the alpha channel if it's present
                image = image[..., :-1]

            # Check if it needs to be converted to greyscale
            if greyscale:
                image = ski.color.gray2rgb(ski.color.rgb2gray(image))

        # Convert the image to ubyte
        image = ski.exposure.rescale_intensity(image, out_range=np.ubyte)

        return image

    def make_colourmap(
        labels: dict | list | tuple,
        cmap: str = "jet",
    ) -> dict:
        """
        Create a dictionary of colours (used for visualising instances).

        Args:
            labels (dict | list | tuple):
                A dictionary of labels.

            cmap (str, optional):
                Colourmap. Defaults to "jet".

        Returns:
            dict:
                Dictionary of class/colour associations.
        """

        if len(labels) == 0:
            return {}

        cmap = plt.get_cmap(cmap, len(labels))
        cmap = cmap(np.linspace(0, 1, cmap.N))[:, :3]
        return {label: colour for label, colour in zip(labels, cmap)}
