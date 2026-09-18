# Unused sources

Source implementations that are no longer part of the `streetscapes` package. They
were moved here because nothing in the package, the CLI or the tests imported them.
They are kept for reference: the Amsterdam Panorama and Global Streetscapes code is
the only record of how those two datasets were accessed.

None of this code runs as-is. All three modules import `streetscapes.sources.base`,
which now lives in this directory rather than in the package, so the imports have to
be fixed up before anything here can be used again.

---

## `base.py`

The class hierarchy the other two modules were built on.

* `SourceBase` — resolves a per-source root directory (`DATA_HOME/sources/<class name>`)
  and reads an access token from a `<CLASS_NAME>_TOKEN` environment variable.
* `ImageSourceBase(SourceBase, ABC)` — adds a `requests` session, an `images` property
  listing downloaded files, `check_image_status` to work out which IDs are still
  missing, and `download_image`.

The live sources (`MapillaryClient`, `KartaViewClient`, `PanoramaxClient`) do not
inherit from these; they are plain classes that the CLI imports directly.

## `amsterdam.py`

`AmsterdamPanorama(ImageSourceBase)`, an interface to the
[Amsterdam panorama API](https://api.data.amsterdam.nl/panorama/panoramas/).

It fetches panorama metadata within a radius of a point, paging through the
`_links.next` chain, and returns an Ibis memtable with `pano_id`, `timestamp`,
position, `heading`/`roll`/`pitch`, and the thumbnail, cubic and equirectangular
image URLs. `get_image_url` raises `NotImplementedError` — the URLs come back with
the metadata instead. The file also carries a commented-out earlier GeoPandas
version of the same fetch.

The intended CLI, which was never wired up:

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

```bash
streetscapes download-images amsterdam \
    manifest.parquet \
    --output-dir images/ \
    --overwrite False
```

* `manifest.parquet`: Manifest file containing `pano_id`s to download.
* `output-dir`: Directory to store downloaded images.
* `overwrite`: Whether to overwrite existing images.

Note that Amsterdam is a centre-plus-radius source, unlike the bounding-box sources
the CLI supports today — which is part of why it never shared their subcommands.

## `global_streetscapes.py`

Access to the [Global Streetscapes](https://ual.sg/project/global-streetscapes/)
dataset on HuggingFace (`NUS-UAL/global-streetscapes`).

* `HFSourceBase(SourceBase, ABC)` — a generic HuggingFace repository interface,
  wrapping `hf_hub_download` with cache inspection (`scan_cache_dir`,
  `try_to_load_from_cache`) so files already in `HF_HUB_CACHE` are reused.
* `GlobalStreetscapesSource(HFSourceBase)` — `load_csv`, `load_parquet` and
  `load_dataset` for the dataset's tables, plus `fetch_image_urls` and
  `dowload_images` (sic), which resolve image URLs through Mapillary and KartaView
  and download from there.

This is the derived-dataset path referred to in the package README's implementation
notes. Deleting it drops the ability to work from Global Streetscapes metadata
altogether; the project still builds on that dataset's results, as the top-level
README and `docs/index.md` describe.

---

## Related notebooks

Two notebooks in `../notebooks/` use these classes:

* `use_data_sources.ipynb` — `AmsterdamPanorama`
* `explore_global_streetscapes.ipynb` — `GlobalStreetscapesSource`

Both are already stale: they import via `from streetscapes.sources import Mapillary,
KartaView, AmsterdamPanorama`, but `sources/__init__.py` exports nothing and there
are no `Mapillary` or `KartaView` classes. The same applies to
`../scripts/Amsterdam/`.
