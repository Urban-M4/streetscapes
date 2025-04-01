# --------------------------------------
from streetscapes.streetview.workspace import SVWorkspace


class SourceBase:

    def __init__(
        self,
        workspace: SVWorkspace,
        name: str | None = None,
    ):

        # Store the workspace
        # ==================================================
        self.workspace = workspace

        # Environment variable prefix
        # ==================================================
        self.name = name.upper() or ""

    def _bootstrap(self):

        # Optional local directory for downloading data from the Global Streetscapes repo.
        # Defaults to the local Huggingface cache directory.
        self.local_dir = self.workspace._env.path(f"{self.name}_LOCAL_DIR", None)
