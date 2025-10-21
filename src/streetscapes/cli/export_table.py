import os

import typer
from streetscapes import config

from streetscapes.project import Project
import logging

logger = logging.getLogger(__name__)

export_cli = typer.Typer(help="Export tables from the project.")


@export_cli.command("table")
def export_table(
    table_name: str = typer.Argument(help="The name of the user to greet"),
    output: str = typer.Argument(help="Output file path"),
):
    """Export a table to (Geo)Parquet, CSV, JSON, GPKG, or GeoJSON."""
    project = Project(config.get("active_project"))
    ext = os.path.splitext(output)[1].lower()

    exporters = {
        ".csv": project.export_csv,
        ".parquet": project.export_parquet,
        ".json": project.export_json,
        ".gpkg": project.export_gpkg,
        ".geojson": project.export_geojson,
    }

    exporter = exporters.get(ext)
    if exporter is None:
        raise typer.BadParameter(f"Unsupported file extension: {ext}")

    exporter(table_name, output)

    logger.info(f"Exported {table_name} to {output}")
