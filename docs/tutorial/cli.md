# Overview
Streetscapes provides a versatile command line interface (CLI) to some of the models that it can work with (currently `BFMS` and [`MaskFormer`](../api/models/maskformer.md); other models to follow shortly). The main entry point (`streetscapes`) is installed together with the package `pip`:

```bash
pip install -e .
```

<!-- TODO: note about installing `tokenizers` and `transformers` *first* -->
If the installation was successful, typing the `streetscapes` command in your terminal should produce a list of available entrypoints:

```bash
streetscapes
```

This should produce the following output:

```bash
Usage: streetscapes COMMAND

Street view image analysis toolkit

╭─ Commands ──────────────────────────────────────────────────────────────────────────────╮
│ config                                                                                  │
│ download_images  Download images from various sources.                                  │
│ export           Export tables from the project.                                        │
│ fetch_metadata   Fetch metadata for a source                                            │
│ segment_images   Segment                                                                │
│ --help (-h)      Display this message and                                               │
│ --version        Display application                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
```

# Subcommands

The functionality of Streetscapes is divided into several categories accessed through subcommands, which are briefly introduced below.

## Configuration

Streetscapes can be configured via the `streetscapes config` command. The current configuration options can be displayed with `streetscapes config list`:

```bash
streetscapes config list

                Streetscapes Configuration
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key            ┃ Value                                     ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ project_dir    │ /<current user>/.local/share/streetscapes │
│ image_dir      │ /<current user>/.cache/streetscapes       │
│ active_project │ streetscapes                              │
└────────────────┴───────────────────────────────────────━━━─┘
```

The `project_dir` directory is where Streetscapes will be storing its projects, databases and output files. The `active_project` will be used by default if a project name is not provided when instantiating a [`Project`](../../src/streetscapes/project.py).

## Downloading images

The Streetscapes CLI supports downloading images from [Mapillary](https://www.mapillary.com/) ([KartaView](https://kartaview.org/landing) and the [Amsterdam](https://api.data.amsterdam.nl/) collection will be added as options shortly). The available options can be displayed with the `--help` option via the subcommand for Mapillary:

<!-- NOTE: This needs to be updated so that we don't need to use `--help` -->
```bash
streetscapes download_images mapillary --help
```

You should see the following output:

```bash
Usage: streetscapes download_images mapillary [ARGS]

Download Mapillary images to a local directory.

╭─ Parameters ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ SKIP-EXISTING --skip-existing --no-skip-existing  If true, only download missing images; otherwise overwrite. [default: True]                                   │
│ TOKEN --token                                     Mapillary OAuth token (if not set via MAPILLARY_TOKEN).                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Note that a [token](https://www.mapillary.com/developer/api-documentation/) is needed to use the Mapillary API.

## Segmenting images

Downloaded images can be segmented with one of several models that offer different feature sets. To view the models currently supported via the CLI, you can run the `segment_images` subcommand:

```bash
streetscapes segment_images
```

You should see the following output:

```bash
Usage: streetscapes segment_images COMMAND

Segment images

╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ bfms        CLI entry point to segment images with BFMS via Ray Serve.                                                │
│ maskformer  Segment images with the MaskFormer model.                                                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Streetscapes uses the [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) library to spawn instances of the segmentation models. Images can be passed to the models via a REST API with dedicated [Pydantic](https://docs.pydantic.dev/latest/concepts/models/) request and response schema defined for each model. The schema and the model service class used for communicating with the actual model are defined in the `service` module for each model (for instance, `models/maskformer/service.py` for the `MaskFormer` model, which is used in the examples below).

The `MaskFormer` model is a wrapper around the [`Mask2Former`](https://huggingface.co/docs/transformers/model_doc/mask2former) model, which is one of the earlier models supporting instance, semantic and panoptic segmentation. `MaskFormer` supports only a limited number of classes (the full list of `65` classes can be accessed via the `id_to_label` attribute of the [`MaskFormer` class](../../src/streetscapes/models/maskformer/model.py)). The full list of options can be viewed via the `--help` argument of the `maskformer` subcommand:

```bash
streetscapes segment_images maskformer --help
```

You should see the following output:

```bash
Usage: streetscapes segment_images maskformer IMAGE-PATH [ARGS]

Segment images with the MaskFormer model.

╭─ Parameters ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  IMAGE-PATH --image-path                        Path to the images to be segmented. [required]                                                                                       │
│    LABELS --labels --empty-labels                 Labels to focus on.                                                                                                                  │
│    BATCH-SIZE --batch-size                        Batch size for the segmentation model. [default: 10]                                                                                 │
│    MODEL-ID --model-id                            Mask2Former model to load. [default: facebook/mask2former-swin-large-mapillary-vistas-panoptic]                                      │
│    THRESHOLD --threshold                          The probability score threshold to keep predicted instance masks. [default: 0.5]                                                     │
│    MASK-THRESHOLD --mask-threshold                Threshold to use when turning the predicted masks into binary values. [default: 0.5]                                                 │
│    OVERLAP-THRESHOLD --overlap-threshold          The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask. [default: 0.8]        │
│    FUSE-LABELS --fuse-labels --empty-fuse-labels  The labels in this state will have all their instances fused together.                                                            │
│    OVERWRITE --overwrite --no-overwrite           Overwrite existing segmentations. [default: False]                                                                                   │
│    BOOTSTRAP --bootstrap --no-bootstrap           (Re)create the model table. [default: False]                                                                                         │
│    PROJECT --project                              The project to use for saving (meta)data. [default: streetscapes]                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Only the `image_path` argument is required. It can be a path to a directory or a single image file. If a directory is passed, all images in that directory will be processed in batches whose size can be specified with the `batch_size` option (the default is `10`). You can also specify a (comma-separated) list of labels (categories of objects) that the model should focus on. By default, if the `--labels` argument is not provided, the model will try to find objects corresponding to ***all*** the categories that it can recognise.

For instance, assuming that you have [downloaded](#downloading-images) some images to a local directory (e.g., `~/data/streetscapes/images`), you can segment all images in that directory with the `MaskFormer` model with the following command:

```bash
streetscapes segment_images maskformer --image_path ~/data/streetscapes/images
```

This command will spawn a `MaskFormer` model using `ray.serve` and will pass images in batches of `10` to the model, which will segment them and return the segmentation masks for all instances of the requested categories. It should produce an output similar to the following:

```bash
[01/13/26 14:30:24] INFO     Failed to auto-detect TPU type.                                                                                                                                                                                                  tpu.py:571
                    INFO     Failed to configure TPU pod. Got: tpu_name: None, worker_id: None, accelerator_type: None                                                                                                                                        tpu.py:630
                    INFO     Failed to auto-detect TPU type.                                                                                                                                                                                                  tpu.py:571
2026-01-13 14:30:24,343 INFO worker.py:1998 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ProxyActor pid=605475) INFO 2026-01-13 14:30:27,270 proxy 192.168.178.186 -- Proxy starting on node d46408fec8cf333c0d185807dbc154f30fb4a023fba2f757d8dab908 (HTTP port: 8000).
INFO 2026-01-13 14:30:27,380 serve 604573 -- Started Serve in namespace "serve".
(ServeController pid=605472) INFO 2026-01-13 14:30:27,450 controller 605472 -- Deploying new version of Deployment(name='ModelApp', app='default') (initial target replicas: 1).
(ProxyActor pid=605475) INFO 2026-01-13 14:30:27,373 proxy 192.168.178.186 -- Got updated endpoints: {}.
(ProxyActor pid=605475) INFO 2026-01-13 14:30:27,457 proxy 192.168.178.186 -- Got updated endpoints: {Deployment(name='ModelApp', app='default'): EndpointInfo(route='/', app_is_cross_language=False, route_patterns=None)}.
(ServeController pid=605472) INFO 2026-01-13 14:30:27,559 controller 605472 -- Adding 1 replica to Deployment(name='ModelApp', app='default').
(ProxyActor pid=605475) INFO 2026-01-13 14:30:27,489 proxy 192.168.178.186 -- Started <ray.serve._private.router.SharedRouterLongPollClient object at 0x7f27e092be90>.
(ServeReplica:default:ModelApp pid=605477) Starting model: maskformer
(ServeReplica:default:ModelApp pid=605477) Streetscapes | 2026-01-13@14:30:31 | Model 'maskformer' using device 'cuda'
INFO 2026-01-13 14:30:35,527 serve 604573 -- Application 'default' is ready at http://127.0.0.1:8000/.
```

Streetscapes stores information about which images have been segmented, using the `SHA-256` hash of the image file instead of the file name to identify images. In this way, the user does not have to worry about renaming or moving files, as long as the images themselves remain unchanged.

Each segmentation model has its own dedicated table in the project database (see the [configuration](#configuration) for options related to projects and storage locations). The segmentation masks produced by the model are saved to individual NumPy archive files with a random `UUID` string as the file name and `.npz` as the extension. The information about which image files have been segmented with which models, and the `UUID`s of the corresponding archive files are saved in the model table. You can use the [`Ibis`](https://ibis-project.org/) library to query the database by launching a Python shell in your terminal:

```bash
python
Python 3.12.12 (main, Dec 17 2025, 21:10:06) [Clang 21.1.4 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
Ctrl click to launch VS Code Native REPL
>>>
```

For instance, you can list the tables the the project database:

```python
>>> import ibis
>>> from streetscapes.project import Project
>>> ibis.options.interactive = True
>>> proj = Project() # This will create an instance of the 'streetscapes' project at the default location.
>>> con = ibis.duckdb.connect(proj.database_path)
>>> con.tables
Tables
------
- bfms
- image_model
- mapillary
- maskformer
```

To check the contents of the model table, use the Ibis connection to list the first five lines of the table:


```python
>>> table = con.table('maskformer')
>>> table.head()
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ uuid                                 ┃ params                                                                                     ┃ timestamp                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ !uuid                                │ !binary                                                                                    │ !timestamp(6)              │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────┤
│ 019b36da-131c-7003-9400-9b9d74fbc485 │ '{\\x22model_id\\x22:\\x22facebook/mask2former-swin-large-mapillary-vistas-panoptic\\'+119 │ 2026-01-12 14:24:06.893866 │
│ 019b36da-1ced-700b-bcd6-9ffab2f17e0e │ '{\\x22model_id\\x22:\\x22facebook/mask2former-swin-large-mapillary-vistas-panoptic\\'+119 │ 2026-01-12 14:24:07.219776 │
│ 019b36da-3603-7004-85ec-14c5ef2acc81 │ '{\\x22model_id\\x22:\\x22facebook/mask2former-swin-large-mapillary-vistas-panoptic\\'+119 │ 2026-01-12 14:24:07.540906 │
│ 019b36da-3c31-7001-9880-8ab62be277e8 │ '{\\x22model_id\\x22:\\x22facebook/mask2former-swin-large-mapillary-vistas-panoptic\\'+119 │ 2026-01-12 14:24:07.829871 │
│ 019b36da-131c-7003-9400-9b9d74fbc485 │ '{\\x22model_id\\x22:\\x22facebook/mask2former-swin-large-mapillary-vistas-panoptic\\'+119 │ 2026-01-13 09:11:19.515741 │
└──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────┘
```

The fields of the table are as follows:

- `uuid`: The name of the archive file (without the `.npz` extension) containing the segmentation masks. The `UUID` is **automatically reused** if the image is segmented with the same model again.
- `params`: The parameters of the model in the form of a binary string (created with `orjson.dumps()`; see the [`orjson`](https://github.com/ijl/orjson) library).
- `timestamp`: The date and time indicating when the image was segmented.

