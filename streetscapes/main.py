import cloup

# --------------------------------------
from pathlib import Path

# --------------------------------------
from streetscapes import conf
from streetscapes.functions import convert_csv_to_parquet


# ==============================================================================
# Main entry point
# ==============================================================================
@cloup.group("streetscapes")
def main():
    return 0


# ==============================================================================
# Preprocessing and conversions
# ==============================================================================
@main.command("convert")
@cloup.option(
    "-d",
    "--data_dir",
    type=Path,
    default=conf.DATA_DIR,
)
@cloup.option(
    "-p",
    "--parquet_filename",
    default="streetscapes-data.parquet",
)
def convert_csv(**kwargs):
    convert_csv_to_parquet(**kwargs)
