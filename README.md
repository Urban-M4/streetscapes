[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)
[![PyPI - Version](https://img.shields.io/pypi/v/streetscapes)](https://pypi.org/project/streetscapes/)
[![Research Software Directory](https://img.shields.io/badge/RSD-streetscapes-00a3e3)](https://research-software-directory.org/software/streetscapes)
[![Read The Docs](https://readthedocs.org/projects/streetscapes/badge/?version=latest)](https://streetscapes.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)

# Streetscapes

`streetscapes` is a Python package, CLI, and web-based explorer for large-scale analysis of street-level imagery.
It bundles functionality ranging from imagery retrieval to segmentation, feature extraction, and building-level aggregation.
The package is designed to be transparent, reproducible, and easy to extend for research use.

## Installation

Requirements: Python 3.14

In a Python virtual environment, or conda environment do:

```bash
pip install streetscapes[sam3]
```

To install for AMD GPUs (ROCm), do;
```bash
uv pip install streetscapes --extra rocm,sam3
```

Retrieving image metadata and downloading images is performed using the command line interface.
For a simple quickstart, see the [tutorial](https://streetscapes.readthedocs.io/en/latest/tutorial/tutorial/) in the documentation.
The documentation also contains more elaborate examples and workflows.

## Contributing and publishing

If you want to contribute to the development of streetscapes,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## 🪪 Licence

`streetscapes` is licensed under [`CC-BY-SA-4.0`](https://creativecommons.org/licenses/by-sa/4.0/deed.en).

## 🎓 Acknowledgements and citation

This repository uses the data and work from the [Global Streetscapes](https://ual.sg/project/global-streetscapes/) project.

> [1] Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T, Biljecki F (2024): Global Streetscapes — A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics. ISPRS Journal of Photogrammetry and Remote Sensing 215: 216-238. doi:[10.1016/j.isprsjprs.2024.06.023](https://doi.org/10.1016/j.isprsjprs.2024.06.023)

The `streetscapes` package can be cited using the supplied [citation information](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files). For reproducibility, you can also cite a specific version by finding the corresponding DOI on [Zenodo](https://zenodo.org/records/14287547).
