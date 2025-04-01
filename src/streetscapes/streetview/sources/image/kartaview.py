# --------------------------------------
from pathlib import Path

# --------------------------------------
from streetscapes.streetview.workspace import SVWorkspace
from streetscapes.streetview.sources.image.base import ImageSourceBase


class KartaviewSource(ImageSourceBase):

    def __init__(
        self,
        workspace: SVWorkspace,
    ):
        super().__init__(
            workspace,
            url=f"https://api.openstreetcam.org/2.0/photo",
            name="KARTAVIEW",
        )

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

            str:
                The URL to query.
        """

        url = f"{self.url}/?id={image_id}"
        try:
            # Send the request
            response = self.session.get(url)

            # Parse the response
            data = response.json()["result"]["data"][0]
            image_url = data["fileurlProc"]
            return image_url

        except Exception as e:
            return
