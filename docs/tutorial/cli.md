# Overview
Streetscapes provides a versatile command line interface (CLI) to perform all common steps (creating a project, retrieving images, segmenting images with computer vision models).
After [installation](../index.md#installing-streetscapes), typing the `streetscapes` command in your terminal should produce a list of available commands:

```bash
streetscapes
```

This should produce the following output:

```bash
Usage: streetscapes COMMAND

Street view image analysis toolkit

╭─ Commands ────────────────────────────────────────────────────────────────────╮
│ config                                                                        │
│ database         Get info and delete entries from the database.               │
│ download-images  Download images from various sources.                        │
│ export           Export tables from the project.                              │
│ fetch-metadata   Fetch metadata for a source                                  │
│ images           Perform various operations on local collections of images.   │
│ segment-images   Segment images                                               │
│ --help (-h)      Display this message and exit.                               │
│ --version        Display application version.                                 │
╰───────────────────────────────────────────────────────────────────────────────╯
```

# Subcommands

The functionality of Streetscapes is divided into several categories accessed through subcommands, which are briefly introduced below.

## Configuration

Streetscapes can be configured via the `streetscapes config` command. The current configuration options can be displayed with `streetscapes config list`:

```bash
streetscapes config list

                Streetscapes Configuration
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key                  ┃ Value                                                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ project_dir          │ /<current user>/.local/share/streetscapes              │
│ image_dir            │ /<current user>/.cache/streetscapes                    │
│ active_project       │ streetscapes                                           │
│ mapillary_token      │ MLY|00000000000000000|00000000000000000000000000000000 │
│ local_cache_dir_name │ local                                                  │
│ sam3_model_path      │ /<SAM3 model dir>/sam3.pt                              │
└──────────────────────┴────────────────────────────────────────────────────────┘
```

The `project_dir` directory is where Streetscapes will be storing its projects, databases and output files. The `active_project` will be used by default if a project name is not provided when instantiating a [`Project`](../../src/streetscapes/project.py).

## Downloading images

The Streetscapes CLI supports downloading images from [Mapillary](https://www.mapillary.com/) ([KartaView](https://kartaview.org/landing) and the [Amsterdam](https://api.data.amsterdam.nl/) collection are currently not yet supported). The available options can be displayed with the `--help` option via the subcommand for Mapillary:

For Mapillary, we first need to fetch image metadata. For this you will need to define a spatial bounding box.
Some areas have enormous amounts of images available. To only get a certain number of images per spatial "tile", set the `--limit` argument.

Note that a [token](https://www.mapillary.com/developer/api-documentation/) is needed to use the Mapillary API.
Register on Mapillary, and register your token with `streetscapes config set mapillary_token YOUR_TOKEN`.

```bash
streetscapes fetch-metadata mapillary --help
```

```bash
Usage: streetscapes fetch-metadata mapillary [OPTIONS] BBOX

Fetch metadata from the Mapillary API.

╭─ Arguments ────────────────────────────────────────────────────────────────────╮
│ *  BBOX  Bounding box (WEST EAST SOUTH NORTH). [required]                      │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ───────────────────────────────────────────────────────────────────╮
│ --tile-size  Tile size in degrees. [default: 0.001]                            │
│ --limit      Maximum number of images per tile. [default: 1000]                │
│ --token      Mapillary OAuth token (if not set via MAPILLARY_TOKEN).           │
│ --project    An optional project to attach to.                                 │
╰────────────────────────────────────────────────────────────────────────────────╯
```

After fetching metadata, you can start the image download:

<!-- NOTE: This needs to be updated so that we don't need to use `--help` -->
```bash
streetscapes download-images mapillary --help
```

You should see the following output:

```bash
Usage: streetscapes download-images mapillary [ARGS]

Download Mapillary images to a local directory.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ SKIP-EXISTING --skip-existing --no-skip-existing  If true, only download missing images; otherwise overwrite. [default: True] │
│ TOKEN --token                                     Mapillary OAuth token (if not set via MAPILLARY_TOKEN).                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Segmenting images

Downloaded images can be segmented with one of several models that offer different feature sets. To view the models currently supported via the CLI, you can run the `segment-images` subcommand:

```bash
streetscapes segment-images
```

You should see the following output:

```bash
Usage: streetscapes segment-images COMMAND

Segment images

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ bfms                                                                         │
│ dinosam                                                                      │
│ maskformer                                                                   │
│ sam3                                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Streetscapes uses the [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) library to spawn instances of the segmentation models. Images can be passed to the models via a REST API with dedicated [Pydantic](https://docs.pydantic.dev/latest/concepts/models/) request and response schema defined for each model. The schema and the model service class used for communicating with the actual model are defined in the `service` module for each model (for instance, `models/maskformer/service.py` for the `MaskFormer` model, which is used in the examples below).

The `MaskFormer` model is a wrapper around the [`Mask2Former`](https://huggingface.co/docs/transformers/model_doc/mask2former) model, which is one of the earlier models supporting instance, semantic and panoptic segmentation. `MaskFormer` supports only a limited number of classes (the full list of `65` classes can be accessed via the `id_to_label` attribute of the [`MaskFormer` class](../../src/streetscapes/models/maskformer/model.py)). The full list of options can be viewed via the `--help` argument of the `maskformer` subcommand:

```bash
streetscapes segment-images maskformer --help
```

You should see the following output:

```bash
Usage: streetscapes segment-images maskformer [ARGS]

Segment images with MaskFormer.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ IMAGE-PATH --image-path                        Path to the images to be segmented. If not provided uses all downloaded images in the project.                             │
│ LABELS --labels --empty-labels                 Labels to focus on.                                                                                                        │
│ BATCH-SIZE --batch-size                        Batch size for the segmentation model. [default: 10]                                                                       │
│ MODEL-ID --model-id                            Mask2Former model to load. [default: facebook/mask2former-swin-large-mapillary-vistas-panoptic]                            │
│ THRESHOLD --threshold                          The probability score threshold to keep predicted instance masks. [default: 0.5]                                           │
│ MASK-THRESHOLD --mask-threshold                Threshold to use when turning the predicted masks into binary values. [default: 0.5]                                       │
│ OVERLAP-THRESHOLD --overlap-threshold          The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask. [default:   │
│                                                0.8]                                                                                                                       │
│ FUSE-LABELS --fuse-labels --empty-fuse-labels  The labels in this state will have all their instances fused together.                                                     │
│ RUN --run                                      Model run ID.                                                                                                              │
│ PROJECT --project                              The project to use. Uses the active project by default. [default: local-test]                                              │
│ OVERWRITE --overwrite --no-overwrite           Overwrite an existing run. [default: False]                                                                                │
│ VERBOSE --verbose --no-verbose                 Print verbose log to the terminal. Useful for debugging models. [default: False]                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

By default, all images in the current streetscapes project will be segmented. These will be processed in batches whose size can be specified with the `batch_size` option (the default is `10`) depending on your hardware it can be better to make that number smaller (laptop) or larger (HPC with GPU). You can also specify a (comma-separated) list of labels (categories of objects) that the model should focus on. By default, if the `--labels` argument is not provided, the model will try to find objects corresponding to ***all*** the categories that it can recognise.

For instance, assuming that you have downloaded images for the current project, use the following command to start segmentation:

```bash
streetscapes segment-images maskformer
```

## Viewing segmentations

Streetscapes provides a browser-based tool for viewing the images and their segmentations. When you enter the following command:

```bash
streetscapes-explorer
```

The explorer backend (serving the project database) will start. This should also open up a web page.
You might be prompted for allowing the page access to your local device, allow this.

To start the explorer on a different port or host, see:

```bash
streetscapes-explorer --help
```