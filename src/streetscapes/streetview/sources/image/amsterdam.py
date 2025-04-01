from streetscapes.streetview.workspace import SVWorkspace
from streetscapes.streetview.sources.image.base import ImageSourceBase


class GoogleSource(ImageSourceBase):

    def __init__(
        self,
        workspace: SVWorkspace,
    ):
        super().__init__(
            workspace,
            url=None,
            name="AMSTERDAM",
        )
