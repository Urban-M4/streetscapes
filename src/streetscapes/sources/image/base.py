# --------------------------------------
import re

# --------------------------------------
from abc import ABC
from abc import abstractmethod

# --------------------------------------
import time

# --------------------------------------
import random

# --------------------------------------
from pathlib import Path

# --------------------------------------
from environs import Env

# --------------------------------------
import requests

# --------------------------------------
from streetscapes import utils
from streetscapes.sources import SourceType
from streetscapes.sources.base import SourceBase


class ImageSourceBase(SourceBase, ABC):

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
            subdir = utils.camel2snake(self.get_source_type().name).lower()
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

        # Mapping of derived source classes
        # ==================================================
        self.sources = {src: set() for src in SourceType}

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
        existing = {path for path in utils.filter_files(self.root_dir, ImageSourceBase.image_pattern) if path.stem in image_ids}
        missing = set(image_ids).difference({img.stem for img in existing})

        return (existing, missing)

    def download_image(
        self,
        image_id: str,
    ) -> Path:
        """
        Download a single image.

        Args:
            image_id:
                The image ID.

        Returns:
            Path:
                The path to the downloaded image file.
        """

        # Set up the image path
        image_path = self.root_dir / f"{image_id}.jpeg"

        # Download the image
        if not image_path.exists():

            # Random sleep time so that we don't flood the servers.
            time.sleep(random.uniform(0.1, 1))

            # Download the image
            # ==================================================
            # NOTE: Specifically in the case of Mapillary,
            # we have to send a request for that image
            # straight after getting the URL.
            # Collecting all the URLs in advance and requesting them
            # one by one outside the loop doesn't work.
            url = self.get_image_url(image_id)
            if url is not None:
                response = self.session.get(url)

                # Save the image if it has been downloaded successfully
                # ==================================================
                if response.status_code == 200 and response.content is not None:
                    with open(image_path, "wb") as f:
                        f.write(response.content)

        return image_path

    def create_session(self) -> requests.Session:
        """
        Create an (authenticated) session for the supplied source.

        Returns:
            A `requests` session.
        """
        return requests.Session()
