import os
from typing import Annotated

import typer
from streetscapes.cli.console import console

from streetscapes.project import Project
import logging

logger = logging.getLogger(__name__)

export_cli = typer.Typer(help="Export tables from the project.")


@export_cli.command("table")
def export_table(
    table_name: str = typer.Argument(help="The name of the user to greet"),
    output: str = typer.Argument(help="Output file path"),
    project_path: str = typer.Option(
        "streetscapes.duckdb", "--project", help="Path to project DB"
    ),
):
    """Export a table to (Geo)Parquet, CSV, JSON, GPKG, or GeoJSON."""
    project = Project(project_path)
    ext = os.path.splitext(output)[1].lower()

    if ext in [".csv"]:
        project.export_csv(table_name, output)
    elif ext in [".parquet"]:
        project.export_parquet(table_name, output)
    elif ext in [".json"]:
        project.export_json(table_name, output)
    elif ext in [".gpkg"]:
        project.export_gpkg(table_name, output)
    elif ext in [".geojson"]:
        project.export_geojson(table_name, output)
    else:
        raise typer.BadParameter(f"Unsupported file extension: {ext}")

    logger.info(f"Exported {table_name} to {output}")
