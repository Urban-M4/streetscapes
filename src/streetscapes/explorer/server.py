"""FastAPI server implementation."""

from contextlib import contextmanager
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Generator, Optional
from uuid import UUID
from itertools import chain

import ibis
import numpy as np
import pandas as pd
import uvicorn
from brotli_asgi import BrotliMiddleware
from cyclopts import App, Parameter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Query
from shapely.ops import transform
from shapely.geometry import Polygon

from streetscapes import CFG
from streetscapes.explorer.data import (
    AggregateStats,
    Bbox,
    FilterParams,
    Image,
    ImageMetadata,
    Instance,
    Segmentation,
)

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(BrotliMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dbpath = Path(f"{CFG.project_dir}/{CFG.active_project}.duckdb")


@contextmanager
def _open_db(dbpath: Path, read_only=True) -> Generator[ibis.BaseBackend, None, None]:
    db = ibis.duckdb.connect(
            dbpath, read_only=read_only, extensions=["spatial", "json"]
        )
    try:
        yield db
    finally:
        db.con.close()


def _get_images(mapillary_table: ibis.Table):
    data = mapillary_table.select("image", "thumb_original_url", "geometry").to_pandas()

    return [
        Image(*args)
        for args in zip(
            data["image"].astype(str),
            data["thumb_original_url"],
            data["geometry"].y,
            data["geometry"].x,
            strict=True,
        )
    ]


def _flip(x, y):
    return y, x


def _get_segmentations(uuid: UUID | str) -> list[Segmentation]:
    with _open_db(dbpath) as con:
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
                    [list(points.exterior.coords) for points in poly.geoms],
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
                instances=inst,
            )
            segmentations.append(seg)
    return segmentations


def _get_metadata(uuid: str) -> ImageMetadata:
    with _open_db(dbpath) as con:
        imgtable = con.table("images")
        imgtable = imgtable.filter(imgtable.uuid == uuid)
        if imgtable.count().to_pandas() == 0:
            raise _unknown_image(uuid)
        imgdata = imgtable.to_pandas().squeeze()

        metatable = con.table(imgdata["source"])
        metatable = metatable.filter(metatable.image == uuid)
        if metatable.count().to_pandas() == 0:
            raise _unknown_image(uuid)
        if metatable.count().to_pandas() > 1:
            metadata = metatable.to_pandas().iloc[0].squeeze()
        else:
            metadata = metatable.to_pandas().squeeze()

    segmentations = _get_segmentations(uuid)
    return ImageMetadata(
        id=str(metadata["image"]),
        url=metadata["thumb_2048_url"],
        lat=metadata["geometry"].y,
        lon=metadata["geometry"].x,
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        altitude=metadata["altitude"],
        captured_at=datetime.fromtimestamp(metadata["captured_at"] / 1000),
        panoramic=bool(metadata["is_pano"]),
        source=imgdata["source"],
        tags=imgdata["tags"],
        rating=0 if imgdata["rating"] in [None, np.nan] else int(imgdata["rating"]),
        compass_angle=float(metadata["compass_angle"]),
        notes="" if imgdata["notes"] in [None, np.nan] else imgdata["notes"],
        segmentation=segmentations,
    )


def _bbox_to_polygon(bbox: Bbox) -> Polygon:
    return Polygon(
        [(bbox.w, bbox.n), (bbox.e, bbox.n), (bbox.e, bbox.s), (bbox.w, bbox.s)]
    )


def _fetch_images(filter: Optional[FilterParams]) -> list[Image]:
    """Fetch images that conform to a filter specification."""
    with _open_db(dbpath) as con:
        if filter is None:
            return _get_images(con.table("mapillary"))  # type: ignore[no-any-return]

        # First filter on image table info
        images = con.table("images")
        if len(filter.image_ratings) > 0:
            match = con.table("images").rating.isin(filter.image_ratings)
            images = images.filter(match)
        if len(filter.sources) > 0:
            match = con.table("images").source.isin(filter.sources)
            images = images.filter(match)
        for tag in filter.tags:
            match = con.table("images").tags.contains(tag)
            images = images.filter(match)

        # Next filter on mapillary table info
        mapillary = con.table("mapillary")
        mapillary = mapillary.filter(mapillary.image.isin(images.uuid))

        if filter.date_range is not None:
            start = filter.date_range[0].timestamp() * 1000
            end = filter.date_range[1].timestamp() * 1000
            match_start = mapillary.captured_at >= start
            match_end = mapillary.captured_at <= end
            mapillary = mapillary.filter(match_start).filter(match_end)

        _match = mapillary.geometry.within(_bbox_to_polygon(filter))
        mapillary = mapillary.filter(_match)

        # Last filter on segmentation properties
        if any(
            (filter.models, filter.labels, filter.model_runs, filter.segmentation_ratings)
        ):
            segmentations = con.table("segmentations")
            # optionally filter for models
            if len(filter.models) > 0:
                runs = con.table("runs")
                runs = runs.filter(runs.model.isin(filter.models))
                segmentations = segmentations.filter(segmentations.run.isin(runs.run))

            for label in filter.labels:
                segmentations = segmentations.filter(segmentations.labels.contains(label))
            if len(filter.model_runs) > 0:
                segmentations = segmentations.filter(
                    segmentations.run.isin(filter.model_runs)
                )
            if len(filter.segmentation_ratings) > 0:
                segmentations = segmentations.filter(
                    segmentations.run.isin(filter.segmentation_ratings)
                )

            valid_images = segmentations.distinct(on="image").image

            _match = mapillary.image.isin(valid_images)
            mapillary = mapillary.filter(_match)

        # Now we can request the valid images
        return _get_images(mapillary)  # type: ignore[no-any-return]


def _update_img_prop(image_id: str, prop: str, value: Any):
    with _open_db(dbpath, read_only=False) as con:
        imgs = con.table("images")
        img = imgs.filter(imgs.uuid == image_id).to_pandas()

        if len(img) == 0:
            _unknown_image(image_id)

        imgd = img.to_dict()
        imgd[prop][0] = value  # workaround for replacing lists
        con.con.register("updated_df", pd.DataFrame(imgd))
        con.raw_sql(f"INSERT OR REPLACE INTO images FROM updated_df;")


def _update_segmentation_rating(image_id: str, run_name: str, rating: int):
    with _open_db(dbpath, read_only=False) as con:
        segs = con.table("segmentations")
        segs = segs.filter(segs.image == image_id)
        seg = segs.filter(segs.run == run_name)

        if len(seg) == 0:
            _unknown_image(image_id)

        imgd = seg.to_dict()
        imgd["rating"][0] = rating
        con.con.register("updated_df", pd.DataFrame(imgd))
        con.raw_sql(f"INSERT OR REPLACE INTO images FROM updated_df;")


def _update_img_prop(image_id: str, prop: str, value: Any):
    with _open_db(dbpath, read_only=False) as con:
        imgs = con.table("images")
        img = imgs.filter(imgs.uuid == image_id).to_pandas()

        if len(img) == 0:
            raise _unknown_image(image_id)

        imgd = img.to_dict()
        imgd[prop][0] = value  # workaround for replacing lists
        con.con.register("updated_df", pd.DataFrame(imgd))
        con.raw_sql(f"INSERT OR REPLACE INTO images FROM updated_df;")


def _update_segmentation_rating(image_id: str, run_name: str, rating: int):
    with _open_db(dbpath, read_only=False) as con:
        segs = con.table("segmentations")
        segs = segs.filter(segs.image == image_id)
        seg = segs.filter(segs.run == run_name)

        if len(seg) == 0:
            raise _unknown_image(image_id)

        imgd = seg.to_dict()
        imgd["rating"][0] = rating
        con.con.register("updated_df", pd.DataFrame(imgd))
        con.raw_sql(f"INSERT OR REPLACE INTO images FROM updated_df;")


def _unknown_image(image_id):
    msg = f"No image found with id '{image_id}'"
    print(msg)
    return HTTPException(status_code=404, detail=msg)


def _get_unique_tags(con: ibis.BaseBackend):
    tags = con.table("images").tags.to_pandas().dropna()
    if len(tags) == 0:
        return []
    tags = set(chain.from_iterable(tags.to_list()))
    return list(tags)


def _get_unique_labels(con: ibis.BaseBackend):
    labels = con.table("segmentations").labels.to_pandas().dropna()
    if len(labels) == 0:
        return []
    labels = set(chain.from_iterable(labels))  # nested list to set of uniques
    return list(labels)


def _get_daterange(con: ibis.BaseBackend) -> tuple[datetime, datetime]:
    # Note: only implemented for mapillary
    with _open_db(dbpath) as con:
        mapillary = con.table("mapillary")
        start = datetime.fromtimestamp(mapillary.captured_at.min().to_pandas() / 1000)
        end = datetime.fromtimestamp(mapillary.captured_at.max().to_pandas() / 1000)
    return (start, end)


def _get_unique_tags(con: ibis.BaseBackend):
    tags = con.table("images").tags.to_pandas().dropna()
    if len(tags) == 0:
        return []
    tags = set(chain.from_iterable(tags.to_list()))
    return list(tags)


def _get_unique_labels(con: ibis.BaseBackend):
    labels = con.table("segmentations").labels.to_pandas().dropna()
    if len(labels) == 0:
        return []
    labels = set(chain.from_iterable(labels))  # nested list to set of uniques
    return list(labels)


def _get_daterange(con: ibis.BaseBackend) -> tuple[datetime, datetime]:
    # Note: only implemented for mapillary
    with _open_db(dbpath) as con:
        mapillary = con.table("mapillary")
        start = datetime.fromtimestamp(mapillary.captured_at.min().to_pandas() / 1000)
        end = datetime.fromtimestamp(mapillary.captured_at.max().to_pandas() / 1000)
    return (start, end)


@app.get("/")
async def root():
    """Server root."""
    return {"message": "Welcome to the Streetscapes Explorer"}


@app.get("/project")
async def project():
    """Get the active project name."""
    return str(CFG.active_project)


@app.get("/stats")
async def fetch_stats() -> AggregateStats:
    """Get the aggregate stats of the images."""
    with _open_db(dbpath) as con:
        return AggregateStats(
            tags=_get_unique_tags(con),
            labels=_get_unique_labels(con),
            model_run_names=list(set(con.table("runs").run.to_pandas())),
            image_sources=list(set(con.table("images")["source"].to_pandas().to_list())),
            date_range=_get_daterange(con),
            models=list(set(con.table("runs").model.to_pandas())),
        )


@app.get("/images")
async def fetch_images(filter: Annotated[FilterParams, Query()]) -> list[Image]:
    """Fetch streetscape images corresponding to a bounding box and optionally filters."""
    return _fetch_images(filter)


@app.get("/images/{image_id}")
async def fetch_image_metadata(image_id: str) -> ImageMetadata:
    """Get all metadata associated with a certain image, including segmentations."""
    try:
        return _get_metadata(image_id)
    except ValueError as err:
        raise _unknown_image(image_id) from err


@app.post("/images/{image_id}/rating")
async def set_rating(image_id: str, rating: int):
    """Set an image's rating."""
    _update_img_prop(image_id, "rating", rating)


@app.post("/images/{image_id}/tags")
async def set_tags(image_id: str, tags: list[str]):
    """Set an image's tags."""
    _update_img_prop(image_id, "tags", tags)


@app.post("/images/{image_id}/notes")
async def set_notes(image_id: str, notes: str):
    """Set an image's notes."""
    _update_img_prop(image_id, "notes", notes)


@app.post("/images/{image_id}/{run_name}/rating")
async def set_segmentation_rating(image_id: str, run_name: str, rating: int):
    """Rate an image's segmentation."""
    _update_segmentation_rating(image_id, run_name, rating)


@app.post("/images/{image_id}/{run_name}/{instance_idx}/{label}")
async def set_instance_label(
    image_id: str, run_name: str, instance_idx: int, label: str
):
    """Set the label of a specific instance within a segmentation."""
    pass


@app.post("/images/{image_id}/segment/{model}/{run_args}")
async def segment_image(image_id, model, run_args):
    """Compute a new segmentation of an image."""
    pass


async def _start_uvicorn(port: int, host: str, log_info: bool):
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info" if log_info else "warning",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _serve(port: int, host: str, open_webpage: bool, log_info: bool):
    server = _start_uvicorn(port, host, log_info)
    if open_webpage:
        print("Waiting for the streetscapes-explorer to start...")
        webbrowser.open(
            f"https://urban-m4.github.io/Urban-M5/?s=http://localhost:{port}"
        )
        print(
            "The streetscapes-explorer should have launched automatically.\n"
            "To open it manually, go to https://urban-m4.github.io/Urban-M5/ and "
        )
    else:
        print(
            "Starting the streetscapes-explorer...\n\n"
            "To open the explorer, go to https://urban-m4.github.io/Urban-M5/ and "
        )
    print(
        f"paste in http://localhost:{port} as web service.\n"
        "  You will need to disable your ad blocker (like uBlock Origin Lite)\n"
        "and allow your web browser to load localhost resources."
    )
    await server


cli = App(help="Streetscapes data explorer")


@cli.default
async def serve(
    *,
    port: Annotated[int, Parameter(name=["--port", "-p"])] = 5001,
    host: Annotated[str, Parameter(name=["--host"])] = "0.0.0.0",
    open_webpage: bool = True,
    verbose_logs: bool = False,
):
    """Start the Streetscapes Explorer server.

    Args:
        port: port to host the backend on.
        host: Bind socket to this host. Default (0.0.0.0) makes the backend available
            to any machine that can communicate with the host. Set it to 127.0.0.1 to
            allow only access from the local machine.
        open_webpage: automatically open a browser window with the frontend viewer,
            with the backend correctly configured.
        verbose_logs: display verbose backend server logs, useful for debugging the
            frontend.
    """
    await _serve(port, host, open_webpage, verbose_logs)


if __name__ == "__main__":
    cli()
