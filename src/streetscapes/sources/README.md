# Streetscapes Metadata Fetching & Image Downloading

This module of the `streetscapes` CLI handles **fetching metadata from image sources** (e.g., Mapillary, Amsterdam Panorama) and **downloading images** based on manifests. Metadata is stored in GeoParquet (or Parquet) format, enabling easy downstream spatial processing with tools like DuckDB or GeoPandas.

These commands are part of the larger `streetscapes` CLI, which also includes image segmentation, visualization, retrieval, building matching, annotation, and model finetuning.

---

## Design Philosophy

* **Atomic and Transparent**: Each source fetch or download is a standalone operation. Researchers can copy a source, manifest writer, or downloader into a new script and run it independently.
* **Separate Commands per Source**: Each source has its own CLI subcommand (`mapillary`, `amsterdam`, etc.) due to differences in API semantics (bounding boxes vs. center+radius, authentication requirements, etc.).
* **Manifest-Driven**: All outputs are manifests (GeoParquet/Parquet) describing images and associated metadata. These can then be used for downstream tasks like downloading images or segmentation.
* **Minimal Complexity**: Avoid hidden initializations or convoluted inheritance. Each step explicitly declares input arguments, so it’s clear what is needed and what is produced.

---

## Metadata Fetching

### Mapillary

```bash
streetscapes fetch-metadata mapillary \
    --bbox W S E N \
    --tile-size 0.01 \
    --output-file mapillary_metadata.parquet \
    --token <MAPILLARY_OAUTH_TOKEN>
```

* `bbox`: Bounding box `[west, south, east, north]` to fetch images from.
* `tile-size`: Optional tiling of the bounding box (default 0.01°).
* `output-file`: Path to save the GeoParquet manifest.
* `token`: OAuth token for Mapillary API.

### Amsterdam Panorama

```bash
streetscapes fetch-metadata amsterdam \
    --lat 52.37 \
    --lon 4.90 \
    --radius 50 \
    --output-file amsterdam_metadata.parquet
```

* `lat`, `lon`: Coordinates for the center of the fetch.
* `radius`: Search radius in meters (default 50m).
* `output-file`: Path to save the GeoParquet manifest.

---

## Image Downloading

### Mapillary

```bash
streetscapes download-images mapillary \
    manifest.parquet \
    --output-dir images/ \
    --overwrite False \
    --token <MAPILLARY_OAUTH_TOKEN>
```

* `manifest.parquet`: Manifest file containing image IDs to download.
* `output-dir`: Directory to store downloaded images.
* `overwrite`: Whether to overwrite existing images.
* `token`: OAuth token for Mapillary API.

### Amsterdam Panorama

```bash
streetscapes download-images amsterdam \
    manifest.parquet \
    --output-dir images/ \
    --overwrite False
```

* `manifest.parquet`: Manifest file containing `pano_id`s to download.
* `output-dir`: Directory to store downloaded images.
* `overwrite`: Whether to overwrite existing images.

---

## Implementation Notes

* **Sources**: Raw sources (Mapillary, KartaView) and derived datasets (e.g., global streetscapes metadata) implement `fetch_metadata` and provide a standardized manifest writer.
* **Manifest Writer**: `PyArrowGeoParquetWriter` ensures output manifests are compatible with spatial operations.
* **Transparency**: Each CLI call is fully self-contained; there are no hidden global states or complicated initialization chains.
* **Extensible**: New sources can be added by implementing `fetch_metadata` and optionally a downloader. The CLI can then expose them as a separate subcommand.

---

## Summary

The metadata and download subcommands provide a **clean, reproducible pipeline** for retrieving and organizing imagery for Streetscapes research and analysis. By separating sources, standardizing outputs, and keeping each step simple and transparent, researchers can focus on experimentation and evaluation without dealing with hidden complexity.
