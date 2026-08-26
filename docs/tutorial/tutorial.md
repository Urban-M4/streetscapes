The main way of working with *streetscapes* is through the command line interface (CLI).
This allows you to download or import images, segment them with CV models, and stores all metadata and segmentations in "DuckDB" databases for efficient access.
It also allows you to start the *streetscapes-explorer*, the browser-based visual interface for evaluating images and segmentations.

The streetscapes CLI is explained [here](cli.md).
However, for custom usecases or analyses not covered by the streetscapes tool, you might want access to the images and segmentations in a programmatic manner.
This is supported with some easy-to-use functions, and is documented in [this tutorial](manual-database-access.ipynb).

## Quickstart

In the tutorial notebooks, we make use of a small, easy-to-test dataset. To reproduce this, make sure you have Python 3.12 (or newer) installed, and have a Mapillary token available.

### Step 1: install streetscapes
The installation is currently from source, but will be on the Python Packaging Index (PyPI) by release.
To install (on a linux system):

```
git clone https://github.com/Urban-M4/streetscapes.git  # clone the repository
cd streetscapes
git checkout dev  # checkout the in-development branch
python3 -m venv .venv  # create a virtual environment
source .venv/bin/activate

pip install transformers  # install required packages
pip install tokenizers

pip install .[explorer]  # Install Streetscapes 
```

### Step 2: configure streetscapes

First we need to set the [Mapillary token](https://www.mapillary.com/developer/api-documentation) so we can fetch metadata and download images from Mapillary:

```                                     
streetscapes config set mapillary_token MLY|00000000000000000|00000000000000000000000000000000  # use your token here
```

Now we can set the project name to the demo project used in the tutorials.

```
streetscapes config set active_project wur-small
```

### Step 3: download images and run a segmentation

```
# fetch data for small area on WUR campus
streetscapes fetch-metadata mapillary --bbox 5.658860 51.982984 5.660480 51.985118 --tile-size 0.001

# download images
streetscapes download-images mapillary

# do segmentation. lower batch size reduces ram usage;
streetscapes segment-images maskformer --batch-size 1
```

### Step 4: (optional) start the explorer to look at the resulting dataset
```
streetscapes-explorer
```
