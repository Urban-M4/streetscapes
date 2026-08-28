# Streetscapes Segmentation

This module implements the **image segmentation** subcommand of the `streetscapes` CLI.
Segmentation is part of a broader toolkit that also includes retrieval of images and metadata, building matching, visualization, annotation, and model fine-tuning.


The goal of this design is to provide a **clean, transparent, and low-complexity architecture** that is accessible both to developers extending the system and to researchers experimenting with models or pipelines.

Currently, we re-implement/wrap various models including DinoSAM, BFMS, the Zero-shot facade material segmentation (ZFMS) pipeline, DMS, ... and (re)combine them in various ways.

## Note on SAM3

Support for the [SAM3 model](https://huggingface.co/facebook/sam3) has been added to StreetScapes. If you plan to use this model, please be aware that it requires **weights to be downloaded manually from the HuggingFace repository**.

---

## 🎯 Design Principles

* **Atomic vs. Composite Models**

  * *Atomic models* are thin wrappers around pretrained vision or segmentation models (e.g. SAM, GroundingDINO, BFMS, CLIPSeg).

    * Input: images as `np.ndarray`.
    * Output: model predictions (masks, bounding boxes, embeddings, etc.), plus a **manifest** describing results per input image. Not all atomic models generate masks or labels (e.g., GroundingDINO returns bounding boxes).
    * No orchestration logic.
  * *Composite models* (e.g. DinoSAM, ZFMS) combine atomic models in sequence.

    * Orchestration is implemented in the **CLI layer**, not in atomic classes.
    * Intermediate outputs are stored to allow inspection, visualization, and debugging.

* **Loose coupling, simple interfaces**

  * Atomic models remain usable standalone (for experimentation, analysis, or fine-tuning).
  * Composite pipelines are user-facing and exposed via the CLI.

* **Consistency for users**

  * All CLI commands behave similarly:

    * Input: single image, folder, list of image paths, or a manifest file.
    * Output: manifest describing results per image, plus any masks or processed outputs.
    * Intermediate outputs (e.g., bounding boxes) are stored automatically for later inspection.

* **Batching and efficiency**

  * Small inputs → processed image by image.
  * Larger inputs → processed in batches (future extensions may add GPU/HPC scheduling).
  * Minimal complexity in batch orchestration; each atomic model handles its own processing.

* **Experimentation-friendly philosophy**

  * Transparent, low-complexity code: easy to understand, modify, or extract into new scripts without having to trace complex inheritance hierarchies or hidden initializations.
  * Easy to understand and modify each step, encouraging reproducibility and experimentation.

---

## 🧩 Architecture Overview

```
streetscapes/
│
├── cli/
│   └── segment_images.py     # Typer CLI for segmentation commands
│
├── segmentation/
│   ├── sam.py                # Atomic: Segment Anything (SAM)
│   ├── groundingdino.py      # Atomic: Grounding DINO (bounding boxes from text prompts)
│   ├── clipseg.py            # Atomic: CLIPSeg (promptable segmentation)
│   ├── openclip.py           # Atomic: OpenCLIP (embeddings/class matching)
│   ├── ade20k.py             # Atomic: ADE20k-based semantic segmentation
│   ├── bfms.py               # Atomic: Mask2Former material segmentation (BFMS)
│   ├── dinosam.py            # Composite: GroundingDINO + SAM (orchestrated in CLI)
│   └── zfms.py               # Composite: multi-step fusion pipeline (CLI orchestrated)
│
└── tests/
    ├── test_models.py        # Unit tests for each atomic model
    └── test_cli.py           # Integration tests for CLI commands
```

---

## 🚀 CLI Usage

Segmentation lives under the `streetscapes segment-images` subcommand.

### Examples

```bash
# Run DinoSAM on a single image
streetscapes segment-images dinosam ./input/image.jpg

# Run BFMS on a folder of images
streetscapes segment-images bfms ./input/images/

# Run MaskFormer on a manifest of image paths
streetscapes segment-images maskformer ./manifests/images.json
```

Each command:

* Runs the selected model (atomic or composite).
* Saves masks, labels, or other model outputs per image.
* Writes a manifest summarizing outputs.
* Stores intermediate results automatically for debugging and evaluation.

---

## 📦 Outputs

Each run produces:

* **Masks or segmentation maps** (if applicable).
* **Bounding boxes or embeddings** (for models like GroundingDINO or OpenCLIP).
* **Manifest file** (JSON or Parquet) recording input/output relationships.
* **Intermediate results** (for composite pipelines) stored for inspection and visualization.

These outputs can later be loaded by the **analysis API** for visualization, evaluation, or further processing.

---

## 🔬 Roadmap

* [ ] Implement atomic models as standalone, minimal wrappers around existing models.
* [ ] Add simple smoke tests for basic functionality (e.g. cli doesn't give error, model runs without issues on blank image)
* [ ] Cleanup from previous implementation(s)
* [ ] Implement composites in CLI (DinoSAM, ZFMS, ...).
* [ ] Add streaming support for very large batches and/or HPC optimizations

---

This design prioritizes **transparency and reproducibility**: each atomic or composite pipeline is explicit, easy to inspect, and easy to extract into a standalone script for experimentation or research.
