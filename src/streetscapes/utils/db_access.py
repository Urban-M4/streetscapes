"""Database access utilities."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Type, overload

import ibis
import numpy as np
import shapely
from PIL import Image
from shapely import Polygon

from streetscapes.config import CFG
from streetscapes.utils.data_structures import Instance, Segmentation

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID


def _flip(coords: np.ndarray) -> np.ndarray:
    """Swap the (y, x) coordinates as stored in the database to (x, y).

    This is applied to all coordinates of a geometry at once. Flipping them one by
    one (with ``shapely.ops.transform``) is about 60 times slower.
    """
    return coords[:, ::-1]


def _validate_rating(rating: Any) -> int:
    if rating is None or np.isnan(rating):
        return 0
    return int(rating)


def _instances_from_row(
    labels: list[str],
    confidences: list[float] | None,
    polygons: Any,
    poly_fmt: Type[str | Polygon],
) -> list[Instance]:
    """Build the instances of a single row of the segmentations table.

    The polygons are stored as (y, x) and are flipped to (x, y) here.
    """
    conf: list[float | None] = (
        [None] * len(labels) if confidences is None else list(confidences)
    )
    multipoly = shapely.transform(polygons, _flip)
    polys = list(multipoly.geoms)
    if len(polys) > len(labels):
        # Older runs stored the background as the first polygon.
        polys.pop(0)
    return [
        Instance(
            label,
            poly
            if poly_fmt == Polygon
            else [list(points.exterior.coords) for points in poly.geoms],
            confidence,
        )
        for label, poly, confidence in zip(labels, polys, conf, strict=True)
    ]


def _run_info(runs: ibis.Table, run: str) -> tuple[str, str]:
    """Return the model name and the (decoded) run arguments of a run."""
    runinfo = runs.filter(runs.run == run).to_pandas()
    if runinfo.empty:
        msg = f"Unknown run ID '{run}'"
        raise ValueError(msg)
    runinfo = runinfo.squeeze()
    meta = runinfo["metadata"]
    if isinstance(meta, str):
        meta = meta.encode("utf8").decode("unicode_escape")
    elif isinstance(meta, dict):
        meta = str(meta)
    return runinfo["model"], meta


@contextmanager
def _reuse_db(
    con: ibis.BaseBackend | None, project: str | None = None
) -> Generator[ibis.BaseBackend, None, None]:
    """Yield the given connection, or open (and close) a read-only one."""
    if con is not None:
        yield con
    else:
        with _open_db(project) as opened:
            yield opened


@contextmanager
def _open_db(
    project: str | None = None, read_only=True
) -> Generator[ibis.BaseBackend, None, None]:
    db = open_project(project, read_only)
    try:
        yield db
    finally:
        db.con.close()


def open_project(
    project: str | None = None,
    read_only: bool = True,
) -> ibis.BaseBackend:
    """Open the database of the currently active project.

    Args:
        project: (optional) name of the project to open.
            Defaults to the active project.
        read_only: (optional) set to False to be able to modify
            the database.
    """
    proj = CFG.active_project if project is None else project
    db_path = (CFG.project_dir / proj).with_suffix(".duckdb")
    return ibis.duckdb.connect(
        db_path,
        extensions=["spatial", "json"],
        read_only=read_only,
    )


def get_image_path(
    uuid: str,
    project: str | None = None,
    err: Callable = lambda msg: ValueError(msg),
) -> Path:
    """Load image based on UUID.

    NOTE: large overlap with _get_image from explorer code.
    """
    db = open_project(project=project)
    imgs = db.table("images")
    imgs = imgs.filter(imgs.uuid == uuid)
    if imgs.count().to_pandas() == 0:
        msg = f"Cannot find image with id '{uuid}'"
        raise err(msg)
    if imgs.count().to_pandas() > 1:
        msg = f"Image id not unique! DB corrupted? id='{uuid}'"
        raise err(msg)
    imgdata = imgs.to_pandas().squeeze()
    file_shard = imgdata["shard"]
    source = imgdata["source"]

    if str(source) == "local":
        file = CFG.image_dir / "images" / str(source) / uuid
    elif file_shard is None:
        msg = "File shard not defined. Cannot find image"
        raise err(msg)
    else:
        file = CFG.image_dir / "images" / str(source) / str(file_shard) / uuid

    if file.with_suffix(".jpg").exists():
        file = file.with_suffix(".jpg")
    elif file.with_suffix(".jpeg").exists():
        file = file.with_suffix(".jpeg")
    else:
        msg = f"Cannot find file at path {file}[.jpg/.jpeg]"
        raise err(msg)

    return file


def get_image(uuid: str) -> Image.Image:
    """Returns an in-memory copy of an image.

    Args:
        uuid: UUID of the image you want to open.
    """
    with Image.open(get_image_path(uuid), mode="r") as im:
        image = im.copy()
    return image


@overload
def get_segmentations(
    uuid: UUID | str,
    project: str | None = None,
    *,
    poly_fmt: Type[str],
) -> list[Segmentation[tuple[float, float]]]: ...


@overload
def get_segmentations(
    uuid: UUID | str,
    project: str | None = None,
    *,
    poly_fmt: Type[Polygon],
) -> list[Segmentation[Polygon]]: ...


def get_segmentations(
    uuid: UUID | str,
    project: str | None = None,
    poly_fmt: Type[str | Polygon] = Polygon,
) -> list[Segmentation]:
    """Get the segmentations of a specific image.

    Args:
        uuid: UUID of the image.
        project: (optional) name of the project. Defaults to the active project.
        poly_fmt: Format of the segmentation's polygons (string or
            shapely Polygon)

    Returns:
        List of segmentation objects.
    """
    with _open_db(project) as con:
        runs = con.table("runs")
        segs = con.table("segmentations")
        seg_data = segs.filter(segs.image == uuid).to_pandas()

        if len(seg_data) < 1:
            return []

        segmentations = []
        for _, row in seg_data.iterrows():
            inst = _instances_from_row(
                row["labels"], row["confidences"], row["polygons"], poly_fmt
            )
            model_name, meta = _run_info(runs, row["run"])
            seg = Segmentation(
                model_name=model_name,
                id=row["run"],
                run_args=meta,
                rating=_validate_rating(row["rating"]),
                instances=inst,
            )
            segmentations.append(seg)
    return segmentations


def _sql_str(value: Any) -> str:
    """Quote a value as an SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def iter_segmentations(
    run: str,
    uuids: Iterable[UUID | str] | None = None,
    batch_size: int = 100,
    project: str | None = None,
    con: ibis.BaseBackend | None = None,
    poly_fmt: Type[str | Polygon] = Polygon,
) -> Generator[dict[str, Segmentation], None, None]:
    """Iterate over the segmentations of a run in batches.

    Loading the segmentations of many images one by one is slow, as every call has to
    open the database and read (and decompress) a large part of the polygons column.
    This function instead reads the segmentations of a whole batch of images in one
    query, which selects a range of row IDs of the segmentations table. DuckDB applies
    such a filter while reading the table, so that only the rows of the batch are
    decompressed. The images are therefore visited in the order in which they are
    stored, not in the order of ``uuids``.

    This pays off when loading a large part of a run. For a handful of images that are
    scattered over the table, every batch still spans a wide range of rows, and calling
    :func:`get_segmentations` per image is just as fast.

    Args:
        run: Run ID of the segmentation run.
        uuids: (optional) UUIDs of the images to load the segmentations of.
            Defaults to all images of the run.
        batch_size: (optional) number of images to load per query. Larger batches need
            considerably more memory (a few GB for several hundred images), and beyond
            a few thousand images a batch exceeds the maximum size of an Arrow buffer.
        project: (optional) name of the project. Defaults to the active project.
        con: (optional) an open database connection to use. Defaults to opening
            (and closing) a read-only connection to the project.
        poly_fmt: Format of the segmentation's polygons (string or shapely Polygon)

    Yields:
        Dictionaries mapping the UUID of an image to its segmentation. An image that
        has no segmentation in this run is left out. If an image has both a curated
        and an uncurated segmentation, the curated one is used.
    """
    with _reuse_db(con, project) as con:
        rows = con.sql(
            f"SELECT image, rowid AS row_id FROM segmentations "
            f"WHERE run = {_sql_str(run)} ORDER BY curated, rowid"
        ).to_pandas()
        row_ids = rows.assign(image=lambda df: df["image"].astype(str)).set_index(
            "image"
        )["row_id"]
        # An image with both a curated and an uncurated segmentation keeps the last
        # (curated) row ID.
        row_ids = row_ids[~row_ids.index.duplicated(keep="last")]
        if uuids is not None:
            wanted = [str(uuid) for uuid in uuids]
            row_ids = row_ids[row_ids.index.isin(wanted)]
        row_ids = row_ids.sort_values()

        runs = con.table("runs")
        model_name, meta = _run_info(runs, run)

        for start in range(0, len(row_ids), batch_size):
            batch = row_ids.iloc[start : start + batch_size]
            # Select the exact rows, as other rows of the same images (of other runs,
            # or uncurated duplicates) can lie within the row ID range. The range is
            # kept, as DuckDB only applies that filter while reading the table.
            ids = ", ".join(str(row_id) for row_id in batch)
            data = con.sql(
                f"SELECT image, labels, confidences, polygons, rating "
                f"FROM segmentations "
                f"WHERE rowid BETWEEN {batch.min()} AND {batch.max()} "
                f"AND rowid IN ({ids})"
            ).to_pandas()
            yield {
                str(row["image"]): Segmentation(
                    model_name=model_name,
                    id=run,
                    run_args=meta,
                    rating=_validate_rating(row["rating"]),
                    instances=_instances_from_row(
                        row["labels"], row["confidences"], row["polygons"], poly_fmt
                    ),
                )
                for _, row in data.iterrows()
            }
