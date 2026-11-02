# Segmentation models

Streetscapes supports several computer vision models for segmenting street view images. All of them are run with the `segment-images` subcommand of the [CLI](cli.md), for example:

```bash
streetscapes segment-images sam3 "building,window,roof,door"
```

The options of each model can be listed with `streetscapes segment-images <model> --help`. The segmentations are stored in the project database, together with the model parameters of the run.

| Model | CLI command | Prompt | Labels |
|-------|-------------|--------|--------|
| [SAM3](#sam3) | `sam3` | Text prompt | Any |
| [DinoSAM](#dinosam) | `dinosam` | Text prompt | Any |
| [MaskFormer](#maskformer) | `maskformer` | None | 65 fixed street scene classes |
| [BFMS](#bfms) | `bfms` | None | Fixed building/facade material classes |

## SAM3

[Segment Anything Model 3](https://ai.meta.com/research/sam3/) by Meta segments all instances of the objects described in a text prompt (e.g. `"building,brick wall,glass wall"`).

!!! note
    SAM3 needs some extra setup:

    1. Install the optional dependencies with `pip install streetscapes[sam3]`. If Ultralytics fails to install CLIP on its own, install it with `pip install git+https://github.com/ultralytics/CLIP.git`.
    2. The model weights can't be downloaded automatically. Request access on the [SAM3 HuggingFace page](https://huggingface.co/facebook/sam3), and download the `sam3.pt` file manually once access is granted.
    3. Set the absolute path to the downloaded weights in the streetscapes config:

        ```bash
        streetscapes config set sam3_model_path "/abs/path/to/sam3.pt"
        ```

## DinoSAM

DinoSAM combines two models, inspired by [LangSAM](https://github.com/luca-medeiros/lang-segment-anything): [GroundingDINO](https://huggingface.co/IDEA-Research/grounding-dino-base) detects the objects described in a text prompt, after which [SAM2](https://huggingface.co/facebook/sam2.1-hiera-large) segments them. The model weights are downloaded automatically from HuggingFace. Other variants (e.g. the smaller `facebook/sam2.1-hiera-tiny`) can be selected with the `--sam-model-id` and `--dino-model-id` options.

## MaskFormer

A wrapper around [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former), using the [model trained on Mapillary Vistas](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-panoptic) by default. It does not take a prompt, but segments a fixed set of 65 street scene classes (e.g. `building`, `road`, `sky`, `vegetation`). The full list is available in the `id_to_label` attribute of the `MaskFormer` class. The model weights are downloaded automatically from HuggingFace.

## BFMS

The [Building/Facade Material Segmentation](https://huggingface.co/jinfengxie/BFMS_1014) model is based on Mask2Former, and segments images into a fixed set of materials (e.g. brick, cement/concrete, glass, wood). It does not take a prompt. The model weights are downloaded automatically from HuggingFace.

!!! note
    BFMS does not support batches: images are segmented one at a time, so it has no `--batch-size` option.
