# --------------------------------------
from abc import ABC
from abc import abstractmethod

# --------------------------------------
import time

# --------------------------------------
import random

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
import os

# --------------------------------------
from pathlib import Path

# --------------------------------------
from environs import Env

# --------------------------------------
import requests

# --------------------------------------
from streetscapes import logger
from streetscapes import utils
from streetscapes.sources.base import SourceBase


class ImageSourceBase(SourceBase, ABC):
    """TODO: Add docstrings"""

    # Regex of extensions for some common image formats.
    # TODO: Parameterise the file extensions.
    # PIL.Image.registered_extensions() is perhaps an overkill.
    image_pattern = r".*(jpe?g|png|bmp|webp|tiff?)"

    def __init__(
        self,
        env: Env,
        root_dir: str | Path | None = None,
        url: str | None = None,
    ):
        """
        A generic interface to an image source.

        Args:

            env:
                An Env object containing loaded configuration options.

            root_dir:
                The directory where images for this source are stored.
                Defaults to None.

            url:
                The base URL of the source. Defaults to None.
        """

        if root_dir is None:
            source_name = self.__class__.__name__.lower()
            subdir = utils.camel2snake(source_name)
            root_dir = utils.create_asset_dir(
                "images",
                subdir,
            )

        super().__init__(env, root_dir)

        # Repository details
        # ==================================================
        self.url = url

        # A session for requesting images
        # ==================================================
        self.session = self.create_session()

    @abstractmethod
    def get_image_url(
        self,
        image_id: int | str,
    ) -> str:
        """
        Retrieve the URL for an image with the given ID.

        Args:
            image_id:
                The image ID.

        Returns:
            The URL to query.
        """
        pass

    def check_image_status(
        self,
        image_ids: set | list | tuple,
    ) -> tuple[set[Path], set[int | str]]:
        """
        Extract the set of IDs for images that have not been downloaded yet.

        Args:
            image_ids:
                A container of image IDs.

        Returns:
            A tuple containing:
                1. A set of paths to existing images.
                2. A set of IDs of missing images (can be used to determine which images to download).
        """

        # Check if images exist.
        # NOTE: This might not be generic.
        # It works for Mapillary and KartaView,
        # but it should be tested on other sources as well.
        image_ids = set([str(r) for r in image_ids])
        existing = {
            path
            for path in utils.filter_files(self.root_dir, ImageSourceBase.image_pattern)
            if path.stem in image_ids
        }
        missing = set(image_ids).difference({img.stem for img in existing})

        return (existing, missing)

    def download_image(
        self,
        image_id: str | int,
        url: str = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Download a single image.

        Args:
            image_id:
                The image ID.

            url:
                The image URL. Defaults to None.

            overwrite:
                Download the image even if it exists.
                Defaults to False.

        Returns:
            Path:
                The path to the downloaded image file.
        """

        # Set up the image path.
        image_path = self.root_dir / f"{image_id}.jpeg"

        # Download the image.
        if image_path.exists() and not overwrite:
            return image_path

        # Download the image
        # ==================================================
        # Fetch the URL if it hasn't been provided.
        if url is None:
            url = self.get_image_url(image_id)

        # Sanity check on the URL.
        if url is None:
            return

        response = self.session.get(url)

        # Save the image if it has been downloaded successfully
        # ==================================================
        if response.status_code == 200 and response.content is not None:
            with open(image_path, "wb") as f:
                f.write(response.content)

        return image_path

    def download_images(
        self,
        image_ids: list[str | int],
        urls: list[str] | None,
    ) -> list[Path]:
        """
        Download a set of images concurrently.

        Args:
            dataset:
                A dataset containing image IDs.

            sample:
                Only download a sample of the images. Defaults to None.

        Returns:
            A list of paths to the saved images.
        """

        # Ensure that we have URLs
        if urls is None:
            urls = [None] * len(image_ids)

        if len(urls) != len(image_ids):
            raise AttributeError("Please ensure that the URL list is the same size as the list of image IDs.")

        # Download the images
        # ==================================================
        results = []
        desc = f"Downloading images | {self.name}"
        with tqdm(total=len(image_ids), desc=desc) as pbar:
            for image_id, url in zip(image_ids, urls):
                try:
                    path = self.download_image(image_id, url)
                    results.append(path)
                    pbar.set_description_str(f"{desc} | {image_id:>20s}")
                except Exception as exc:
                    logger.info(f"Failed to download image {image_id}:\n{exc}")
                pbar.update()
            pbar.set_description_str(f"{desc} | Done")

        return results

    def create_session(self) -> requests.Session:
        """
        Create an (authenticated) session for the supplied source.

        Returns:
            A `requests` session.
        """
        return requests.Session()
