# --------------------------------------
from pathlib import Path

# --------------------------------------
import random

# --------------------------------------
import time

# --------------------------------------
import ibis

# --------------------------------------
import requests

# --------------------------------------
from huggingface_hub import cached_assets_path

# --------------------------------------
from streetscapes.streetview.workspace import SVWorkspace
from streetscapes.streetview.sources.base import SourceBase


class ImageSourceBase(SourceBase):

    def __init__(
        self,
        workspace: SVWorkspace,
        url: str | None,
        name: str | None = None,
    ):

        super().__init__(workspace, name)

        # Repository details
        # ==================================================
        self.url = url
        self.token = workspace._env(f"{self.name}_TOKEN", None)
        self.directory = self.workspace.create_image_dir(self.name.lower())

        # A session for requesting images
        # ==================================================
        self.session = self.create_session()

        # Bootstrap the source
        # ==================================================
        self._bootstrap()

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

        raise NotImplementedError("Please override this method in a derived class.")

    def download_image(
        self,
        image_id: int,
    ) -> Path:
        """
        Download a single image from Mapillary.

        Args:
            image_id:
                The image ID.

        Returns:
            The path to the downloaded image file.
        """

        raise NotImplementedError("Please override this method in a derived class.")

    def get_missing_image_ids(
        self,
        records: ibis.Table,
        existing: set[Path] = None,
    ) -> tuple[set[Path], set[int | str]]:
        """
        Extract the set of IDs for images that have not been downloaded yet.

        Args:
            records:
                An Ibis table containing image ID information.

            existing:
                A set of paths that has already been collected.
                Defaults to None.

        Returns:
            A tuple containing:
                1. A set of paths to existing images.
                2. A set of image IDs to download.
        """

        if existing is None:
            existing = set()

        missing = set()
        for records in records:
            image_id = records["orig_id"]
            image_path = self.directory / f"{image_id}.jpeg"
            if image_path.exists():
                existing.add(image_path)
            else:
                missing.add(image_id)

        return (existing, missing)

    def download_image(
        self,
        image_id: int,
    ) -> Path:
        """
        Download a single image from Mapillary.

        Args:
            image_id:
                The image ID.

        Returns:
            Path:
                The path to the downloaded image file.
        """

        # Set up the image path
        image_path = self.directory / f"{image_id}.jpeg"

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
