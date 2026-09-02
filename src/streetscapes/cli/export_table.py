"""Export table CLI."""

import logging
import os

from cyclopts import App

logger = logging.getLogger(__name__)

export_cli = App(help="Export tables from the project.")


@export_cli.command(name="table")
def export_table(
    table_name: str,
    output: str,
    /,
    *,
    project: str | None = None,
):
    """Export a table to (Geo)Parquet, CSV, JSON, GPKG, or GeoJSON.

    Args:
        table_name: The name of the table to export.
        output: Output file path (must have .csv, .parquet, .json, .gpkg, or
            .geojson extension).
        project: Optionally specify the project to work on.
    """
    from streetscapes.project import Project

    proj = Project(project)
    ext = os.path.splitext(output)[1].lower()

    exporters = {
        ".csv": proj.export_csv,
        ".parquet": proj.export_parquet,
        ".json": proj.export_json,
        ".gpkg": proj.export_gpkg,
        ".geojson": proj.export_geojson,
    }

    exporter = exporters.get(ext)
    if exporter is None:
        raise ValueError(f"Unsupported file extension: {ext}")

    exporter(table_name, output)
    logger.info(f"Exported {table_name} to {output}")
