[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)
[![PyPI - Version](https://img.shields.io/pypi/v/streetscapes)](https://pypi.org/project/streetscapes/)
[![Research Software Directory](https://img.shields.io/badge/RSD-streetscapes-00a3e3)](https://research-software-directory.org/software/streetscapes)
[![Read The Docs](https://readthedocs.org/projects/streetscapes/badge/?version=latest)](https://streetscapes.readthedocs.io/en/latest/)

# Overview

Streetscapes is a package to extract metadata, download, segment and analyse street view images from various open sources, such as [Mapillary](https://www.mapillary.com/), [Kartaview](https://kartaview.org/landing) and [Amsterdam Open Panorama](https://amsterdam.github.io/projects/open-panorama/). 'streetscapes' also builds upon the [Global Streetscapes](https://ual.sg/project/global-streetscapes/), making it possible to download and use the dataset analysis or downloading images with certain properties. 

'streetscapes' is a subproject of ([Urban-M4](https://github.com/Urban-M4)), which aims to model the Urban Heat Island effect by evaluating the properties of individual objects in the images (such as buildings, roads and sidewalks) that could help estimate the amount of heat trapped in urban environments.

For more information, please refer to the [documentation](https://streetscapes.readthedocs.io/en/latest/).

## 📥 Setup

Create and activate a virtual environment using the tool of your choice, such as [venv](https://docs.python.org/3/library/venv.html). You can also use [Conda](https://anaconda.org/) (or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)) if you prefer, but please note that all dependencies are installed by `pip` from `PyPI`.

Using 'venv':

```sh
python -m venv .venv
source .venv/bin/activate
```

Using 'conda':

```sh
conda create -n myenv -c conda-forge python=3.12 pip
conda activate myenv
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

⚠️ Installing `streetscapes` is necessary in order to run any of the example notebooks.

⚠️ If one or more dependencies fail to install, check the Python version - it might be too _new_. While `streetscapes` itself specifies only the _minimal_ required Python verion, some dependencies might be slow to make releases for the latest Python version.

### 🌲 Environment variables

To facilitate the use of `streetscapes` for different local setups, some environment variables can be added to an `.env` file in the root directory of the `streetscapes` repository. Note that the mapillary token is (only) needed if you are using imagery from Mapillary. You can get it [here](https://www.mapillary.com/developer/api-documentation).

| Variable                  | Description                                                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MAPILLARY_TOKEN`         | A Mapillary token string used for authentication when querying Mapillary via their API.  |


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

## Contributing and publishing

If you want to contribute to the development of streetscapes,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## 🪪 Licence

`streetscapes` is licensed under [`CC-BY-SA-4.0`](https://creativecommons.org/licenses/by-sa/4.0/deed.en).

## 🎓 Acknowledgements and citation

This repository uses the data and work from the [Global Streetscapes](https://ual.sg/project/global-streetscapes/) project.

> [1] Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

The `streetscapes` package can be cited using the supplied [citation information](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files). For reproducibility, you can also cite a specific version by finding the corresponding DOI on [Zenodo](https://zenodo.org/records/14287547).
