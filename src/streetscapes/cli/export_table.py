import logging
import os

from cyclopts import App

from streetscapes import config

logger = logging.getLogger(__name__)

export_cli = App(help="Export tables from the project.")


@export_cli.command(name="table")
def export_table(
    table_name: str,
    output: str,
):
    """Export a table to (Geo)Parquet, CSV, JSON, GPKG, or GeoJSON.

    Parameters
    ----------
    table_name:
        The name of the table to export.
    output:
        Output file path (must have .csv, .parquet, .json, .gpkg, or .geojson extension).
    """
    from streetscapes.project import Project

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
        raise ValueError(f"Unsupported file extension: {ext}")

    exporter(table_name, output)
    logger.info(f"Exported {table_name} to {output}")
