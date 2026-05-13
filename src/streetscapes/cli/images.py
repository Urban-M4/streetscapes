"""CLI commands to view/manipulate the database."""

from pathlib import Path
from cyclopts import App

image_cli = App(help="Perform various operations on local collections of images.")


@image_cli.command(name="add")
def add_images(
    path: Path,
    project: str | None = None,
    shard: str | None = None,
    overwrite: bool = False,
):
    """
    Add images from a local directory.

    Args:
        path: A path to a local directory containing images.
        project: The name of the project to use for these images.
        shard: Optional path shard (to a subpath for the images).
        overwrite: Overwrite images that have already been added.
    """

    from streetscapes.project import Project

    proj = Project(project)
    proj.add_local_images(path, shard, overwrite)
