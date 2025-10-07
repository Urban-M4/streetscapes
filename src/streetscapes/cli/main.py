import typer
from dotenv import load_dotenv

from .fetch_metadata import fetch_metadata_cli
from .finetune_model import finetune_model_cli

# from .download_images import download_images_cli
from .identify_buildings import identify_buildings_cli
from .export_table import export_cli
from .segment_images import segment_images_cli

load_dotenv()

app = typer.Typer(help="Street view image analysis toolkit")

# Add subcommand groups

app.add_typer(export_cli, name="export")
app.add_typer(fetch_metadata_cli, name="fetch_metadata")
# app.add_typer(download_images_cli, name="download_images")
# app.add_typer(segment_images_cli, name="segment_images")
# app.add_typer(finetune_model_cli, name="finetune_model")
# app.add_typer(identify_buildings_cli, name="identify_buildings")

if __name__ == "__main__":
    app()
