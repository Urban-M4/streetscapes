# `streetscapes`
This repository contains information and code for retrieving and using data from the [global-streetscapes](https://github.com/ualsg/global-streetscapes/tree/main) dataset.

# Installation
Clone the repository, enter the root directory and use pip to install the `streetscapes` package:

```shell
$> git clone git@github.com:Urban-M4/streetscapes.git
<...>
$> cd streetscapes
$> pip install -e .
```

### Dependencies
There are a lot more dependencies in `pyproject.toml` than strictly necessary to run the examples in this repository. They are necessary for running (at least part of) the code in the original `global-streetscapes` repository, specifically the training pipeline (WIP).

Streetscapes uses a [custom version](https://github.com/Urban-M4/mapillary-python-sdk) of the [Mapillary Python SDK](https://github.com/mapillary/mapillary-python-sdk) which fixes some dependency issues.

## Environment variables
To facilitate usage across different setups, some environment variables can can be added to an `.env` file (`<repo-root>` is the root directory of the `streetscapes` repository):

- `MAPILLARY_CLIENT_ID`: A Mapillary client ID string (can be obtained via your [Mapillary account](https://www.mapillary.com/app/)).
- `MAPILLARY_TOKEN`: A Mapillary token string used for authentication when querying Mapillary via their API.
- `STREETSCAPES_DATA_PATH`: A directory containing data from the `global-streetscapes` projects, such as CSV files (cf. below). Defaults to `<repo-root>/local/streetscapes-data`.
- `STREETSCAPES_OUTPUT_DIR`: A directory for output files. Defaults to `<repo-root>/local/output`.

## CLI
Streetscapes provides a command line interface (CLI) that exposes some of the internal functions. To get the list of available commands, run the CLI with the `--help` switch:

```shell
$> streetscapes --help
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

## Examples and analysis
Currently, there are several notebooks (located under `<repo-root>/notebooks`) demonstrating how to work with the dataset.

### Notebooks
- `plot_city.ipynb`: Shows a simple of example of subsetting the dataset and plotting the data.
- `subset_data.ipynb`: Shows an example of subsetting the data for image download, similar to [this example](https://github.com/ualsg/global-streetscapes/blob/main/code/download_imgs/sample_subset_download.ipynb).
- `mapillary.ipynb`: Shows an example of how to download and display images from Mapillary.

### Acknowledgements/Citation
This repository uses the data and work from:

[1] Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

