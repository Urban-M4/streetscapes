The main way of working with *streetscapes* is through the command line interface (CLI).
This allows you to download or import images, segment them with CV models, and stores all metadata and segmentations in "DuckDB" databases for efficient access.
It also allows you to start the *streetscapes-explorer*, the browser-based visual interface for evaluating images and segmentations.

The streetscapes CLI is explained [here](tutorial.md).
However, for custom usecases or analyses not covered by the streetscapes tool, you might want access to the images and segmentations in a programmatic manner.
This is supported with some easy-to-use functions, and is documented in [this tutorial](manual-database-access.ipynb).
