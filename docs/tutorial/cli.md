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
│ config           View and modify the streetscapes configuration.              │
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

The Streetscapes CLI supports downloading images from [Mapillary](https://www.mapillary.com/), [KartaView](https://kartaview.org/landing) and [Panoramax](https://panoramax.fr/) (the [Amsterdam](https://api.data.amsterdam.nl/) collection is currently not yet supported). All three follow the same two steps — fetch the metadata for a bounding box, then download the images it describes. The available options can be displayed with the `--help` option via the subcommand for each source:

### Mapillary

For Mapillary, we first need to fetch image metadata. For this you will need to define a spatial bounding box.
Some areas have enormous amounts of images available. To only get a certain number of images per spatial "tile", set the `--tile-limit` argument.

Note that a [token](https://www.mapillary.com/developer/api-documentation/) is needed to use the Mapillary API.
Register on Mapillary, and register your token with `streetscapes config set mapillary_token YOUR_TOKEN`.

```bash
streetscapes fetch-metadata mapillary --help
```

```bash
Usage: streetscapes fetch-metadata mapillary [OPTIONS] BBOX

Fetch metadata from the Mapillary API.

╭─ Arguments ────────────────────────────────────────────────────────────────────╮
│ *  BBOX  Bounding box (WEST SOUTH EAST NORTH). [required]                      │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ───────────────────────────────────────────────────────────────────╮
│ --tile-size   Tile size in degrees. [default: 0.001]                           │
│ --tile-limit  Maximum number of images per tile. [default: 1000]               │
│ --token       Mapillary OAuth token (if not set via MAPILLARY_TOKEN).          │
│ --project     An optional project to attach to.                                │
╰────────────────────────────────────────────────────────────────────────────────╯
```

After fetching metadata, you can start the image download:

<!-- NOTE: This needs to be updated so that we don't need to use `--help` -->
```bash
streetscapes download-images mapillary --help
```

You should see the following output:

```bash
Usage: streetscapes download-images mapillary [OPTIONS]

Download Mapillary images to a local directory.

╭─ Parameters ─────────────────────────────────────────────────────────────────╮
│ --skip-existing       If true, only download missing images; otherwise       │
│   --no-skip-existing  overwrite. [default: True]                             │
│ --token               Mapillary OAuth token (if not set via                  │
│                       MAPILLARY_TOKEN).                                      │
│ --project             An optional project to attach to.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### KartaView

KartaView works the same way, but needs no token, and its API pages through a
bounding box of any size, so there is no tiling and `--image-limit` caps the number
of images for the whole bounding box (use `--image-limit 0` to fetch all of them):

```bash
streetscapes fetch-metadata kartaview --help
```

```bash
Usage: streetscapes fetch-metadata kartaview [OPTIONS] BBOX

Fetch metadata from the KartaView API.

╭─ Arguments ────────────────────────────────────────────────────────────────────╮
│ *  BBOX  Bounding box (WEST SOUTH EAST NORTH). [required]                      │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ───────────────────────────────────────────────────────────────────╮
│ --image-limit  Maximum number of images to fetch (0 for no limit). [default:   │
│                1000]                                                           │
│ --project      An optional project to attach to.                               │
╰────────────────────────────────────────────────────────────────────────────────╯
```

The images are then downloaded in the same way:

```bash
streetscapes download-images kartaview
```

KartaView serves its images at full resolution, which for recent cameras means 4K.
Segmenting those needs a correspondingly large amount of memory, because the models
scale their masks back up to the size of the image they were given — so on a machine
with limited RAM, segment KartaView images in small batches (`--batch-size 1`).

### Panoramax

Panoramax is federated: rather than one central server, it is a network of
instances that each host their own pictures. Streetscapes queries the
[federated catalogue](https://docs.panoramax.fr/federated-catalog/) by default,
which indexes every instance taking part. Pass `--instance` to search just one
instance, or one that is not federated:

```bash
streetscapes fetch-metadata panoramax 2.34 48.856 2.345 48.86 \
    --instance https://panoramax.openstreetmap.fr
```

No token is needed for downloading the images.
As for Mapillary, the bounding box is split into tiles and
`--tile-limit` caps the images fetched per tile — but Panoramax returns up to
32767 images per request against Mapillary's ~2000, so its tiles can be far
larger before running into problems.

```bash
streetscapes fetch-metadata panoramax --help
```

```bash
Usage: streetscapes fetch-metadata panoramax [OPTIONS] BBOX

Fetch metadata from the Panoramax API.

Queries the federated catalogue by default, which indexes every Panoramax instance
taking part in the federation, so one query covers them all. Pass --instance to
search a single instance instead.

The API returns no more than 32767 images per request and offers no paging, so the
bounding box is split into tiles, as it is for Mapillary. Panoramax tiles can be
much larger than Mapillary's, as its limit is far higher.

╭─ Arguments ────────────────────────────────────────────────────────────────────╮
│ *  BBOX  Bounding box (WEST SOUTH EAST NORTH). [required]                      │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ───────────────────────────────────────────────────────────────────╮
│ --tile-size   Tile size in degrees. [default: 0.05]                            │
│ --tile-limit  Maximum number of images per tile (at most 32767, which is also  │
│               what 0 means: the most the API will return). [default: 1000]     │
│ --instance    A single Panoramax instance to query, such as                    │
│               'https://panoramax.openstreetmap.fr'. Defaults to the federated  │
│               catalogue.                                                       │
│ --project     An optional project to attach to.                                │
╰────────────────────────────────────────────────────────────────────────────────╯
```

Downloading is straightforward with the following command:

```bash
streetscapes download-images panoramax
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
│ bfms        Segment images with BFMS.                                        │
│ dinosam     Segment images with DinoSAM.                                     │
│ maskformer  Segment images with MaskFormer.                                  │
│ sam3        Segment images with SAM3.                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Streetscapes uses the [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) library to spawn instances of the segmentation models. Images can be passed to the models via a REST API with dedicated [Pydantic](https://docs.pydantic.dev/latest/concepts/models/) request and response schema defined for each model. The schema and the model service class used for communicating with the actual model are defined in the `service` module for each model (for instance, `models/maskformer/service.py` for the `MaskFormer` model, which is used in the examples below).

The `MaskFormer` model is a wrapper around the [`Mask2Former`](https://huggingface.co/docs/transformers/model_doc/mask2former) model, which is one of the earlier models supporting instance, semantic and panoptic segmentation. `MaskFormer` supports only a limited number of classes (the full list of `65` classes can be accessed via the `id_to_label` attribute of the [`MaskFormer` class](../../src/streetscapes/models/maskformer/model.py)). The full list of options can be viewed via the `--help` argument of the `maskformer` subcommand:

```bash
streetscapes segment-images maskformer --help
```

You should see the following output:

```bash
Usage: streetscapes segment-images maskformer [OPTIONS]

Segment images with MaskFormer.

╭─ Parameters ─────────────────────────────────────────────────────────────────╮
│ --image-path             Path to the images to be segmented. If not provided │
│                          uses all downloaded images in the project.          │
│ --batch-size             Batch size for the segmentation model. [default:    │
│                          10]                                                 │
│ --model-id               Mask2Former model to load. [default:                │
│                          facebook/mask2former-swin-large-mapillary-vistas-pa │
│                          noptic]                                             │
│ --threshold              The probability score threshold to keep predicted   │
│                          instance masks. [default: 0.5]                      │
│ --mask-threshold         Threshold to use when turning the predicted masks   │
│                          into binary values. [default: 0.5]                  │
│ --overlap-threshold      The overlap mask area threshold to merge or discard │
│                          small disconnected parts within each binary         │
│                          instance mask. [default: 0.8]                       │
│ --fuse-labels            The labels in this state will have all their        │
│   --empty-fuse-labels    instances fused together.                           │
│ --run                    Model run ID. Will be generated automatically if    │
│                          not provided.                                       │
│ --project                The project to use. Uses the active project by      │
│                          default. [default: local-test]                      │
│ --overwrite              Overwrite an existing run. [default: False]         │
│ --verbose                Print verbose log to the terminal. Useful for       │
│                          debugging models. [default: False]                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

By default, all images in the current streetscapes project will be segmented. These will be processed in batches whose size can be specified with the `batch_size` option (the default is `10`) depending on your hardware it can be better to make that number smaller (laptop) or larger (HPC with GPU).

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