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
* `pano-only`: Only fetch panoramic images. The API filters on `is_pano` itself, so
  `tile-limit` counts panoramas only.
* `token`: OAuth token for Mapillary API.

### KartaView

```bash
streetscapes fetch-metadata kartaview \
    W S E N \
    --image-limit 1000
```

* `bbox`: Bounding box `[west, south, east, north]` to fetch images from.
* `image-limit`: Maximum number of images to fetch for the bounding box (`0` for
  no limit). The KartaView API pages through a bounding box of any size, so there
  is no tiling, and the limit is not per tile as it is for Mapillary.
* `pano-only`: Only fetch panoramic images, i.e. those whose `projection` is not
  `PLANE`. The API cannot filter on this, so the listing is filtered as it comes
  in and paged through until `image-limit` panoramas are found — in an area with
  few of them, that can mean listing the whole bounding box.

No token is required. Metadata is collected in two steps, as no single public
endpoint both covers a bounding box and returns complete records: the photos in
the box are listed first, then their full records are fetched in batches by ID.

### Panoramax

```bash
streetscapes fetch-metadata panoramax \
    W S E N \
    --tile-size 0.05 \
    --tile-limit 1000 \
    --instance https://panoramax.openstreetmap.fr
```

* `bbox`: Bounding box `[west, south, east, north]` to fetch images from.
* `tile-size`: Tiling of the bounding box, in degrees (default 0.05°). Tiles can
  be much larger than Mapillary's, as Panoramax returns far more per request.
* `tile-limit`: Maximum number of images per tile, at most 32767 — which is also
  what `0` means, the most the API will return.
* `instance`: A single Panoramax instance to query. Optional — defaults to the
  federated catalogue.
* `pano-only`: Only fetch panoramic images, i.e. those with a 360° field of view.
  The federated catalogue filters on this itself, but single instances reject the
  filter, so with `--instance` the results are filtered afterwards and
  `tile-limit` applies before the other images are dropped.

No token is required. Panoramax is a federation of instances rather than one
server, so by default the [federated catalogue](https://docs.panoramax.fr/federated-catalog/)
at `https://api.panoramax.xyz` is queried, which indexes every participating
instance; `--instance` restricts the search to one of them, or reaches one that is
not federated. The search endpoint is STAC and offers no paging, so 32767 images is
the most one request can yield; a fetch that hits the limit says so.

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

### KartaView

```bash
streetscapes download-images kartaview \
    --skip-existing
```

* `skip-existing`: Only download the images that are missing.

Images are downloaded at full resolution, which for recent cameras means 4K.

### Panoramax

```bash
streetscapes download-images panoramax \
    --skip-existing
```

* `skip-existing`: Only download the images that are missing.

No instance is given here: each picture is downloaded from whichever instance
hosts it, which is recorded in the `instance` column when the metadata is fetched.
Much of Panoramax is 360° imagery at up to 8000×4000.

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

* **Sources**: Raw sources (Mapillary, KartaView, Panoramax) and derived datasets (e.g., global streetscapes metadata) implement `fetch_metadata` and provide a standardized manifest writer.
* **Manifest Writer**: `PyArrowGeoParquetWriter` ensures output manifests are compatible with spatial operations.
* **Transparency**: Each CLI call is fully self-contained; there are no hidden global states or complicated initialization chains.
* **Extensible**: New sources can be added by implementing `fetch_metadata` and optionally a downloader. The CLI can then expose them as a separate subcommand.

---

## Summary

The metadata and download subcommands provide a **clean, reproducible pipeline** for retrieving and organizing imagery for Streetscapes research and analysis. By separating sources, standardizing outputs, and keeping each step simple and transparent, researchers can focus on experimentation and evaluation without dealing with hidden complexity.
