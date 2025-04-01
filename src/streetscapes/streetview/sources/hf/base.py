from streetscapes.streetview.sources.base import SourceBase
from streetscapes.streetview.workspace import SVWorkspace


class HFSourceBase(SourceBase):

    def __init__(
        self,
        workspace: SVWorkspace,
        repo_id: str,
        repo_type: str,
        env_prefix: str | None = None,
    ):

        super().__init__(workspace, env_prefix)

        # Repository details
        # ==================================================
        self.repo_id = repo_id
        self.repo_type = repo_type

        # Bootstrap the source
        # ==================================================
        self._bootstrap()
