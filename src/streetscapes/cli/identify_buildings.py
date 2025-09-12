import typer

identify_buildings_cli = typer.Typer(
    help="Match buildings in images to building IDs from a GIS database."
)


@identify_buildings_cli.command("generate_mapping")
def generate_mapping(images: list[str], building_footprints: str):
    """
    Generate mapping from image segments to building IDs.

    Args:
        images (list[str]): List of file paths to input images.
        building_footprints (str): File path to GIS database of building footprints.
    """
    # Placeholder for actual finetuning logic
    typer.echo("Generating mapping from image segments to building IDs")
    raise NotImplementedError("Mapping generation not yet implemented")
