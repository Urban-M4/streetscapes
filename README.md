[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)
![PyPI - Version](https://img.shields.io/pypi/v/streetscapes)
[![Research Software Directory](https://img.shields.io/badge/RSD-streetscapes-00a3e3)
](https://research-software-directory.org/software/streetscapes)

# `streetscapes`
This repository contains information and code for retrieving and using data from the [global-streetscapes](https://github.com/ualsg/global-streetscapes/tree/main) dataset.

# Installation
1. Create a virtual environment

Use [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/stable/) or a wrapper such as [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to create a virtual environment. A barebones `environment.yml` file is provided for convenience in case you prefer to use [Conda](https://anaconda.org/) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), but please note that all dependencies are installed by `pip` from `PyPI`.

2. Install streetscapes from `PyPI`

```shell
$> pip install streetscapes
```

Streetscapes requires a custom version of `mapillary-python-sdk`:

```shell
$> pip install git+https://github.com/Urban-M4/mapillary-python-sdk.git
```

4. Set up environment variables

To facilitate the use of `streetscapes` for different local setups, some environment variables can be added to an `.env` file in the root directory of the `streetscapes` repository.

- `MAPILLARY_TOKEN`: A Mapillary token string used for authentication when querying Mapillary via their API.
- `STREETSCAPES_DATA_DIR`: A directory containing data from the `global-streetscapes` projects, such as CSV files (cf. below). Defaults to `<repo-root>/local/streetscapes-data`.
- `STREETSCAPES_OUTPUT_DIR`: A directory for output files. Defaults to `<repo-root>/local/output`.
- `STREETSCAPES_LOG_LEVEL`: The global log level. Defaults to `INFO`.

### Dependencies
There are a lot more dependencies in `pyproject.toml` than strictly necessary to run the examples in this repository. They are necessary for running (at least part of) the code in the original `global-streetscapes` repository, specifically the training pipeline (WIP).

Streetscapes uses a [custom version](https://github.com/Urban-M4/mapillary-python-sdk) of the [Mapillary Python SDK](https://github.com/mapillary/mapillary-python-sdk) which fixes some dependency issues.

## CLI
Streetscapes provides a command line interface (CLI) that exposes some of the internal functions. To get the list of available commands, run the CLI with the `--help` switch:

```shell
$> streetscapes --help

Usage: streetscapes [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  convert
  download
```

For instance, CSV files from the `global-streetscapes` project can be converted into `parquet` format with the CLI as follows:

```shell
$> streetscapes convert
```

The `convert_csv_to_parquet` function inside `streetscapes.functions` contains the code for reproducing the merged `streetscapes-data.parquet` dataset. This function expects a directory  containing several CSV files, which can be downloaded from [Huggingface](https://huggingface.co/datasets/NUS-UAL/global-streetscapes/tree/main/data). The code looks for these csv files in the supplied directory, which defaults to `./local/streetscapes-data` but can be changed with the `-d` switch (cf. `streetscapes convert --help`). Nonexistent directories are created automatically.

To limit the size of the archive, the dataset currently combines the following CSV files:

- `contextual.csv`
- `metadata_common_attributes.csv`
- `segmentation.csv`
- `simplemaps.csv`

It is possible to combine more CSV files if needed.

More CLI commands will be added as the codebase grows.

## Examples and analysis
Currently, there are several notebooks (located under `<repo-root>/notebooks`) demonstrating how to work with the dataset.

### Notebooks
- `plot_city.ipynb`: Shows a simple of example of subsetting the dataset and plotting the data.
- `subset_data.ipynb`: Shows an example of subsetting the data for image download, similar to [this example](https://github.com/ualsg/global-streetscapes/blob/main/code/download_imgs/sample_subset_download.ipynb).
- `mapillary.ipynb`: Shows an example of how to download and display images from Mapillary.

### Acknowledgements/Citation
This repository uses the data and work from:

[1] Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

