from cyclopts import App
from dotenv import load_dotenv

from streetscapes.cli.config import config_cli
from streetscapes.cli.download_images import download_images_cli
from streetscapes.cli.export_table import export_cli
from streetscapes.cli.fetch_metadata import fetch_metadata_cli
from streetscapes.cli.segment_images import segment_images_cli
from streetscapes.cli.database import database_cli
from streetscapes.cli.images import image_cli

# from streetscapes.cli.finetune_model import finetune_model_cli
# from streetscapes.cli.identify_buildings import identify_buildings_cli

load_dotenv()

app = App(help="Street view image analysis toolkit")

# Add subcommand groups

app.command(config_cli)
app.command(database_cli, name="database")
app.command(image_cli, name="images")
app.command(export_cli, name="export")
app.command(fetch_metadata_cli, name="fetch-metadata")
app.command(download_images_cli, name="download-images")
app.command(segment_images_cli, name="segment-images")
# app.add(finetune_model_cli, name="finetune_model")
# app.add(identify_buildings_cli, name="identify_buildings")

if __name__ == "__main__":
    app()
