# --------------------------------------
import cloup

# --------------------------------------
import pandas as pd

# --------------------------------------
from pathlib import Path

# --------------------------------------
from streetscapes import conf
from streetscapes.functions import convert_csv_to_parquet
from streetscapes.functions import download_images


# ==============================================================================
# Main entry point
# ==============================================================================
@cloup.group("streetscapes")
def main():
    return 0


# ==============================================================================
# Preprocessing and conversion
# ==============================================================================
@main.command("convert")
@cloup.option(
    "-d",
    "--data_dir",
    type=Path,
    default=conf.DATA_DIR,
)
@cloup.option(
    "-o",
    "--out_dir",
    type=Path,
    default=conf.OUTPUT_DIR,
)
@cloup.option(
    "-f",
    "--filename",
    default="streetscapes-data.parquet",
)
@cloup.option(
    "-s",
    "--silent",
    is_flag=True,
    default=False,
)
def convert_csv(**kwargs):
    convert_csv_to_parquet(**kwargs)


@main.command("download")
@cloup.option_group(
    "Input file (separate switches by type)",
    cloup.option(
        "-c",
        "--csv",
    ),
    cloup.option(
        "-p",
        "--parquet",
    ),
    constraint=cloup.constraints.mutually_exclusive,
)
@cloup.option(
    "-d",
    "--directory",
    type=Path,
    default=conf.OUTPUT_DIR,
)
@cloup.option(
    "-s",
    "--sample",
    is_flag=True,
    default=False,
)
@cloup.option(
    "-r",
    "--resolution",
    type=int,
    default=2048,
)
def download_images_from_file(**kwargs):

    for fmt in ('csv', 'parquet'):
        path = kwargs.pop(fmt)
        if path is not None:
            df = pd.read_csv(path)
    download_images(df, **kwargs)
