# Convert maps to WRF input

The albedo and emissivity maps produced in the examples (e.g.
[facade albedo](../examples/ams-albedo.ipynb) and
[facade emissivity](../examples/ams-emissivity.ipynb)) are saved as GeoTIFF files.
To use them in WRF, they first have to be converted to the
[WPS geogrid binary format](https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html#writing-static-data-to-the-geogrid-binary-format),
so that `geogrid.exe` can interpolate them onto the model domain.

The repository contains a standalone script for this,
[`scripts/geotiff_to_wps.py`](https://github.com/Urban-M4/streetscapes/blob/main/scripts/geotiff_to_wps.py).
It declares its own dependencies, so you can run it with `uv` without installing
streetscapes:

```sh
uv run scripts/geotiff_to_wps.py docs/examples/building_albedo.tiff wps_input/building_albedo
uv run scripts/geotiff_to_wps.py docs/examples/facade_emissivity.tiff wps_input/facade_emissivity
```

Each output directory contains an `index` file and a single data tile, e.g.:

```
wps_input/building_albedo/
├── 00001-00032.00001-00031
└── index
```

## What the script does

- **Reprojection:** geogrid only supports a few projections. Input that is not in
  lat/lon (for example the Dutch RD New grid, EPSG:28992) is reprojected to a regular
  lat/lon grid (EPSG:4326, `projection = regular_ll`) using nearest-neighbour
  resampling, so no new values are invented. The lat/lon grid spacing is chosen
  separately for each direction to match the source resolution, so no source cells
  are dropped.
- **Missing data:** NaN or nodata cells are written as `missing_value = -999`.
- **Scaling:** values are stored as 2-byte signed integers. They are divided by the
  scale factor and rounded; geogrid multiplies by `scale_factor` again when reading.
  The default of `0.001` gives three decimals of precision for values between
  -32.768 and 32.767, which is suitable for albedo and emissivity.
- **Description:** taken from the GeoTIFF band description, unless overridden.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--scale-factor` | `0.001` | Multiplier applied by geogrid to the stored integers. |
| `--units` | `NO UNITS` | Units written to the index file. |
| `--description` | band description | Description written to the index file. |

## Using the data in WPS

Add an entry for each field to `GEOGRID.TBL`, pointing `abs_path` (or `rel_path`)
to the output directory, for example:

```
===============================
name = BUILDING_ALBEDO
        priority = 1
        dest_type = continuous
        fill_missing = -999
        interp_option = default:average_gcell(4.0)+four_pt+average_4pt
        abs_path = /path/to/wps_input/building_albedo/
===============================
```

Then run `geogrid.exe` as usual; the field will appear in the `geo_em.d0*.nc` files.
Model grid points outside the data are set to the `fill_missing` value.

Use `interp_option = default:nearest_neighbor` instead if you want the original
values to be copied without interpolation.
