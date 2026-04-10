"""CLI commands to view/manipulate the database."""

from pathlib import Path
from typing import TYPE_CHECKING

from cyclopts import App

from rich.table import Table
from rich.console import Console

from streetscapes import CFG

if TYPE_CHECKING:
    import ibis

image_cli = App(help="Perform various operations on local collections of images.")


@image_cli.command(name="add")
def add_images(
    path: Path,
    project: str | None = None,
    shard: str | None = None,
    overwrite: bool = False,
):
    """Add images from a local directory."""
    from streetscapes.project import Project

    proj = Project(project)
    proj.add_local_images(path, shard, overwrite)
