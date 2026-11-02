# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "rasterio",
# ]
# ///
"""Convert a single-band GeoTIFF to WPS geogrid binary format (index + tile).

Based on https://github.com/Urban-M4/WRF/issues/5, extended with:
- reprojection to EPSG:4326 (geogrid's `regular_ll`) for non-lat/lon inputs
- NaN nodata handling
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

MISSING_VALUE = -999
DTYPE = "<i2"  # little endian, signed integer, wordsize 2
WGS84 = CRS.from_epsg(4326)


def read_as_latlon(path):
    """Return (data, transform) on a regular lat/lon grid, NaN for missing values."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        if src.crs == WGS84:
            return data, src.transform

        # Match the source resolution along each axis. The default (square degrees)
        # is coarser than the source in the north-south direction, dropping cells.
        left, bottom, right, top = transform_bounds(src.crs, WGS84, *src.bounds)
        resolution = ((right - left) / src.width, (top - bottom) / src.height)
        transform, width, height = calculate_default_transform(
            src.crs, WGS84, src.width, src.height, *src.bounds, resolution=resolution
        )
        dest = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            data,
            dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=WGS84,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        return dest, transform


def convert(path, outdir, description, scale_factor, units):
    """Write the WPS binary tile and index file for `path` to `outdir`."""
    data, transform = read_as_latlon(path)
    ny, nx = data.shape

    scaled = np.where(np.isnan(data), MISSING_VALUE, np.round(data / scale_factor))
    info = np.iinfo(DTYPE)
    valid = scaled[scaled != MISSING_VALUE]
    if valid.size and (valid.min() < info.min or valid.max() > info.max):
        raise ValueError("Scaled values exceed int16 range; use a larger scale factor")

    outdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / f"{1:05d}-{nx:05d}.{1:05d}-{ny:05d}"
    scaled.astype(DTYPE).tofile(filename)

    # Center of the top-left cell; with row_order = top_bottom this is (1, ny)
    lon, lat = transform * (0.5, 0.5)
    dx = transform.a
    dy = -transform.e

    index = f"""projection = regular_ll
known_x    = 1
known_y    = {ny}
known_lat  = {lat:.6f}
known_lon  = {lon:.6f}
dx         = {dx:.5e}
dy         = {dy:.5e}
tile_x     = {nx}
tile_y     = {ny}
tile_z     = 1
type = continuous
signed = yes
wordsize = 2
endian = little
row_order = top_bottom
tile_bdr = 0
missing_value = {MISSING_VALUE}
scale_factor = {scale_factor}
units = "{units}"
description = "{description}"
"""
    (outdir / "index").write_text(index)
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geotiff", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--description", default=None)
    parser.add_argument("--scale-factor", type=float, default=0.001)
    parser.add_argument("--units", default="NO UNITS")
    args = parser.parse_args()

    description = args.description
    if description is None:
        with rasterio.open(args.geotiff) as src:
            description = src.descriptions[0] or args.geotiff.stem

    filename = convert(
        args.geotiff, args.outdir, description, args.scale_factor, args.units
    )
    print(f"Wrote {filename} and {args.outdir / 'index'}")
