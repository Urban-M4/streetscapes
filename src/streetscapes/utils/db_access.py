"""Database access utilities."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Type, overload

import ibis
import numpy as np
from PIL import Image
from shapely import Polygon
from shapely.ops import transform

from streetscapes.config import CFG
from streetscapes.utils.data_structures import Instance, Segmentation

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID


def _flip(x, y):
    return y, x


def _validate_rating(rating: Any) -> int:
    if rating is None or np.isnan(rating):
        return 0
    return int(rating)


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

    if file_shard is None:
        msg = "File shard not defined. Cannot find image"
        raise err(msg)

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
            labels = row["labels"]
            multipoly = transform(_flip, row["polygons"])  # type: ignore[arg-type]
            polys = list(multipoly.geoms)
            if len(polys) > len(labels):
                polys.pop(0)
            inst = [
                Instance(
                    label,
                    poly
                    if poly_fmt == Polygon
                    else [list(points.exterior.coords) for points in poly.geoms],
                )
                for label, poly in zip(labels, polys, strict=True)
            ]
            runinfo = runs.filter(runs.run == row["run"]).to_pandas().squeeze()
            meta = runinfo["metadata"]
            if isinstance(meta, str):
                meta = meta.encode("utf8").decode("unicode_escape")
            elif isinstance(meta, dict):
                meta = str(meta)
            seg = Segmentation(
                model_name=runinfo["model"],
                id=row["run"],
                run_args=meta,
                rating=_validate_rating(row["rating"]),
                instances=inst,
            )
            segmentations.append(seg)
    return segmentations
