[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)
[![PyPI - Version](https://img.shields.io/pypi/v/streetscapes)](https://pypi.org/project/streetscapes/)
[![Research Software Directory](https://img.shields.io/badge/RSD-streetscapes-00a3e3)](https://research-software-directory.org/software/streetscapes)
[![Read The Docs](https://readthedocs.org/projects/streetscapes/badge/?version=latest)](https://streetscapes.readthedocs.io/en/latest/)

# Overview

Streetscapes is a project within a project. The aim of the parent project ([Urban-M4](https://github.com/Urban-M4)) is to model the Urban Heat Island effect with the help of various software tools. As part of the `Urban-M4` ecosystem, Streetscapes provides a streamlined API for downloading, segmenting and analysing street view imagery (SVI). Specifically, the analysis is geared towards evaluating the properties of individual objects in the images (such as buildings, roads and sidewalks) that could help estimate the amount of heat trapped in urban environments. Streetscapes is based on and builds upon the results of the [Global Streetscapes](https://ual.sg/project/global-streetscapes/) project.

This repository contains information and code for downloading, segmenting and analysing images from Mapillary and KartaView, using information from [global-streetscapes](https://github.com/ualsg/global-streetscapes/tree/main) dataset.

For more information, plese refer to the [documentation](https://streetscapes.readthedocs.io/en/latest/).

## 📥 Setup

Use [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/stable/) or a wrapper such as [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) or [uv](https://docs.astral.sh/uv/pip/environments/) to create a virtual environment. You can also use [Conda](https://anaconda.org/) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) if you prefer, but please note that all dependencies are installed by `pip` from `PyPI`.

```sh
# Create a virtual environment using your tool of preference
python -m venv venv
uv venv venv
virtualenv venv
mkvirtualenv venv
conda create -n myenv -c conda-forge python=3.12 pip
mamba create -n myenv -c conda-forge python=3.12 pip

# Dont forget to activate the environment once created, e.g. source venv/bin/activate or mamba activate myenv
```

### ⚙️ Installation

The `streetscapes` package can be installed from PyPI:

```shell
pip install streetscapes
```

Alternatively, `streetscapes` can be installed by cloning the repository and installing the package locally with `pip`:

```shell
git clone git@github.com:Urban-M4/streetscapes.git
cd streetscapes
pip install -e .
```

Some of the dependencies in `pyproject.toml` are there in anticipation for replicating and extending the pipelines in the original `global-streetscapes` repository, specifically the training pipeline (currently work in progress).

⚠️ Installing `streetscapes` is necessary in order to run any of the example notebooks.

⚠️ If one or more dependencies fail to install, check the Python version - it might be too _new_. While `streetscapes` itself specifies only the _minimal_ required Python verion, some dependencies might be slow to make releases for the latest Python version.

### 🌲 Environment variables

To facilitate the use of `streetscapes` for different local setups, some environment variables can be added to an `.env` file in the root directory of the `streetscapes` repository. Note that the mapillary token is (only) needed if you are using imagery from Mapillary. You can get it [here](https://www.mapillary.com/developer/api-documentation).

| Variable                  | Description                                                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MAPILLARY_TOKEN`         | A Mapillary token string used for authentication when querying Mapillary via their API.  |
| `STREETSCAPES_DATA_DIR`   | A directory containing data from the `global-streetscapes` projects, such as CSV files (cf. below). Defaults to `<repo-root>/local/streetscapes-data`.       |
| `STREETSCAPES_OUTPUT_DIR` | A directory for output files. Defaults to `<repo-root>/local/output`.                                                                                        |
| `STREETSCAPES_LOG_LEVEL`  | The global log level. Defaults to `INFO`.                                                                                                                    |

The `.gitignore` file already contains entries for `.env` and `local/`, so they will never be committed.

### 📖 (Optional) Documentation

#### Installing

⚠️ To work on the documentation locally, please install `streetscapes` with the `dev` optional dependencies:

```shell
pip install streetscapes[dev]
```

or, if installing from source,

```shell
pip install -e .[dev]
```

#### Building and running

The `streetscapes` project documentation is based on [MkDocs](https://www.mkdocs.org/). To compile the documentation, run `mkdocs build` from the `docs` directory:

```shell
cd docs
mkdocs build
```

The documentation can then be viewed locally by starting the `MkDocs` server (again, from the `docs` directory):

```shell
mkdocs serve
```

This will start an HTTP server which can be accessed by visiting `http://127.0.0.1:8000` in a browser.

If building and serving the documentation from a different directory, the location of the `mkdocs.yml` file relative to the current working directory must be specified explicitly. For instance, if running from the root directory of the repository:

```shell
mkdocs build -f docs/mkdocs.yml
mkdocs serve -f docs/mkdocs.yml
```

## Downloading streetscapes metadata

This package is designed around the Global Streetscapes dataset [available on Hugging Face](https://huggingface.co/datasets/NUS-UAL/global-streetscapes)

We can use the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/index) package to download the data. This package was installed as part of the streetscapes package. It can be used both in a Python script/notebook or on the command line. For example, to download the contextual data file:

```sh 
 huggingface-cli download NUS-UAL/global-streetscapes data/contextual.csv --repo-type dataset --local-dir ./local/streetscapes-data
```

Hugging Face hub downloads data to a cache directory by default, but it is possible to specify a local directory, which we have done here to align with the environment variables set above (legacy).

Datasets will be downloaded or loaded from cache as needed by the streetscapes package. However, if you prefer, you could pre-download the full dataset in one go:

```sh 
 huggingface-cli download NUS-UAL/global-streetscapes --repo-type dataset --local-dir ./local/streetscapes-data
```

⚠️ Note, this amounts to 64GB of disk space!


## Python interface for exploratory analysis

<<<<<<< HEAD
This package support both an exploratory workflow through a Python API, and a command line interface for more consolidated workflow. We recommend to start by checking out the [example notebooks](https://streetscapes.readthedocs.io/en/latest/).
=======
This package supports both an exploratory workflow through a Python API, and a command line interface for a more consolidated workflow. We recommend to start by running the [example notebooks](https://streetscapes.readthedocs.io/en/latest/).
>>>>>>> origin/main

## ⌨️ Command-line interface for consolidated workflows

Streetscapes provides a command-line interface (CLI) that exposes some of the internal functions. To get the list of available commands, run the CLI with the `--help` switch:

```shell
streetscapes --help

Usage: streetscapes [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  convert
  download
```

For instance, CSV files from the `global-streetscapes` project can be converted into `parquet` format with the CLI as follows:

```shell
streetscapes convert
```

⚠️ Don't forget to [set up your environment variables](#environment-variables) if using the CLI.

The `convert_csv_to_parquet` function inside `streetscapes.functions` contains the code for reproducing the merged `streetscapes-data.parquet` dataset. This function expects a directory containing several CSV files, which can be downloaded from [Huggingface](https://huggingface.co/datasets/NUS-UAL/global-streetscapes/tree/main/data). The code looks for these csv files in the supplied directory, which defaults to `./local/streetscapes-data` but can be changed with the `-d` switch (cf. `streetscapes convert --help`). Nonexistent directories are created automatically.

To limit the size of the archive, the dataset currently combines the following CSV files:

- `contextual.csv`
- `metadata_common_attributes.csv`
- `segmentation.csv`
- `simplemaps.csv`

It is possible to combine more CSV files if needed.

More CLI commands will be added as the codebase grows.

## Contributing and publishing

If you want to contribute to the development of streetscapes,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## 🪪 Licence

`streetscapes` is licensed under [`CC-BY-SA-4.0`](https://creativecommons.org/licenses/by-sa/4.0/deed.en).

## 🎓 Acknowledgements and citation

This repository uses the data and work from the [Global Streetscapes](https://ual.sg/project/global-streetscapes/) project.

> [1] Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

The `streetscapes` package can be cited using the supplied [citation information](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files). For reproducibility, you can also cite a specific version by finding the corresponding DOI on [Zenodo](https://zenodo.org/records/14287547).
