[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14283584.svg)](https://doi.org/10.5281/zenodo.14283533)
[![PyPI - Version](https://img.shields.io/pypi/v/streetscapes)](https://pypi.org/project/streetscapes/)
[![Research Software Directory](https://img.shields.io/badge/RSD-streetscapes-00a3e3)](https://research-software-directory.org/software/streetscapes)
[![Read The Docs](https://readthedocs.org/projects/streetscapes/badge/?version=latest)](https://streetscapes.readthedocs.io/en/latest/)

# Overview

Streetscapes is a project within a project. The aim of the parent project ([Urban-M4](https://github.com/Urban-M4)) is to model the Urban Heat Island effect with the help of various software tools. As part of the `Urban-M4` ecosystem, Streetscapes provides a streamlined API for downloading, segmenting and analysing street view imagery (SVI). Specifically, the analysis is geared towards evaluating the properties of individual objects in the images (such as buildings, roads and sidewalks) that could help estimate the amount of heat trapped in urban environments. Streetscapes is based on and builds upon the results of the [Global Streetscapes](https://ual.sg/project/global-streetscapes/) project.

This repository contains information and code for downloading, segmenting and analysing images from Mapillary and KartaView, using information from [global-streetscapes](https://github.com/ualsg/global-streetscapes/tree/main) dataset.

## Installing streetscapes

The streetscapes python package is available on PyPI. Install this in a Python 3.14 environment with:

```
pip install streetscapes[sam3,explorer]
```

To install streetscapes with support for AMD GPUs (ROCm), do;

```
pip install uv  # if you don't have uv yet
uv pip install streetscapes==1.0.0a0 --extra rocm,sam3,explorer
```

For more information on the CLI, see [this page](tutorial/cli.md).

## Segmentation models

Streetscapes supports several computer vision models for segmenting the images (SAM3, DinoSAM, MaskFormer and BFMS). See [segmentation models](tutorial/models.md) for an overview, and for the extra setup some models need.

## Examples

- [Facade classification in Amsterdam](examples/ams-map-workflow.ipynb): classify building facades as brick, concrete or glass with SAM3, and map the facade fractions on a grid.
- [Facade classification with confidence](examples/ams-map-workflow-confidence.ipynb): a variant of the facade classification where overlapping facade masks are resolved by the most confident SAM3 instance.
- [Facade emissivity map](examples/ams-emissivity.ipynb): convert the facade fractions into a map of facade emissivity, written to a GeoTIFF file.
- [Building albedo](examples/ams-albedo.ipynb): estimate building albedo from the HSV value of the building pixels, and write it to a GeoTIFF file on the same grid as the emissivity map.
- [Building color analysis](examples/ams-building-color.ipynb): compute the mean building color per image, and show how building color and hue vary over the city.
