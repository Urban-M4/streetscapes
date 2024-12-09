[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)
[![PyPI - Version](https://img.shields.io/pypi/v/streetscapes)](https://pypi.org/project/streetscapes/)
[![Research Software Directory](https://img.shields.io/badge/RSD-streetscapes-00a3e3)
](https://research-software-directory.org/software/streetscapes)

# Introduction
This repository contains information and code for downloading, segmenting and analysing images from Mapillary and KartaView, using information from [global-streetscapes](https://github.com/ualsg/global-streetscapes/tree/main) dataset.

# Setup
1. Create a virtual environment

Use [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/stable/) or a wrapper such as [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to create a virtual environment. A barebones `environment.yml` file is provided for convenience in case you prefer to use [Conda](https://anaconda.org/) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), but please note that all dependencies are installed by `pip` from `PyPI`.

2. Install streetscapes from `PyPI`

```shell
$> pip install streetscapes
```

3. et up environment variables

To facilitate the use of `streetscapes` for different local setups, some environment variables can be added to an `.env` file in the root directory of the `streetscapes` repository.

- `MAPILLARY_TOKEN`: A Mapillary token string used for authentication when querying Mapillary via their API. Only needed if you are using imagery from Mapillary.
- `STREETSCAPES_DATA_DIR`: A directory containing data from the `global-streetscapes` projects, such as CSV files (cf. below). Defaults to `<repo-root>/local/streetscapes-data`.
- `STREETSCAPES_OUTPUT_DIR`: A directory for output files. Defaults to `<repo-root>/local/output`.
- `STREETSCAPES_LOG_LEVEL`: The global log level. Defaults to `INFO`.

4. (Optional) Build the documentation

The `streetscapes` project documentation is built with [MkDocs](https://www.mkdocs.org/). To compile the documentation, run the following command from the root directory of the repository:

```shell
$> mkdocs build -f docs/mkdocs.yml
```

The documentation also contains several notebooks demonstrating how to work with the datasets and the library. The notebooks can also be downloaded and run locally.

### Dependencies
There are a more dependencies in `pyproject.toml` than strictly necessary to run the examples in this repository. This is in preparation for running (at least part of) the code in the original `global-streetscapes` repository, specifically the training pipeline (currently work in progress).

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

### Acknowledgements/Citation
This repository uses the data and work from:

[1] Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

The `streetscapes` package can be cited using the supplied [citation information](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files). For reproducibility, you can also cite a specific version by finding the corresponding DOI on [Zenodo](https://zenodo.org/records/14287547).