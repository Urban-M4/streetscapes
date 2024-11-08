# streetscapes
This repository contains information and code for retrieving and using data from the [global-streetscapes](https://github.com/ualsg/global-streetscapes/tree/main) dataset.

### Converting global-streetscapes to parquet

`convert_csv_to_parquet.py` contains the code for reproducing `streetscapes-data.parquet` dataset.
For input you will have to download the original csv files from [hugging face](https://huggingface.co/datasets/NUS-UAL/global-streetscapes/tree/main/data). The code looks for these csv files in `../streetscapes-data`, which you may need to create yourself or change the path to the location of the files.
To limit size, the dataset contains the following data:

    - `contextual.csv`
    - `metadata_common_attributes.csv`
    - `segmentation.csv`
    - `simplemaps.csv`

It is possible to combine more CSV files if needed.

### Analysing the data

`plot_city.ipynb` shows a simple of example of subsetting the dataset and plotting the data.
`subset_data.ipynb` shows an example of subsetting the data for image download, similar to [this example](https://github.com/ualsg/global-streetscapes/blob/main/code/download_imgs/sample_subset_download.ipynb)

## Installation
Just clone the repository and run

`$> pip install -e .`

from the project root.

### Dependencies
There are a lot more dependencies in `pyproject.toml` than strictly necessary to run the examples in this repository. They are necessary for running (at least part of) the code in the original `global-streetscapes` repository, specifically the training pipeline (WIP).

### Acknowledgements/Citation

This repository uses the data and work from:

Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)
