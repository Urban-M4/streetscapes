"""FastAPI server implementation."""

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID
from itertools import chain

import ibis
import pandas as pd
import uvicorn
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
from streetscapes.explorer.dummy_data import _images

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
dbpath = Path(f"{CFG.project_dir}/{CFG.active_project}.duckdb")
con: ibis.BaseBackend = ibis.duckdb.connect(dbpath, read_only=True, extensions=["spatial", "json"])


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


def _check_result_count(data: pd.DataFrame, id: str | UUID) -> None:
    if len(data) < 1:
        msg = f"No entry found with ID {id}"
        raise ValueError(msg)
    if len(data) > 1:
        msg = f"More than one entry found with ID {id}. Database corrupted?"


def _get_source(uuid: UUID) -> str:
    match = con.table("images").uuid == uuid
    matching_images = con.table("images").filter(match).select("source")
    result = matching_images.to_pandas()
    _check_result_count(result, uuid)
    return result.values[0][0]


def _flip(x, y):
    return y, x


def _get_segmentations(uuid: UUID) -> list[Segmentation]:
    runs = con.table("runs")
    segs = con.table("segmentations")
    seg_filter = segs.image == uuid
    seg_data = segs.filter(seg_filter).to_pandas()

    if len(seg_data) < 1:
        msg = f"No segmentations found with ID {id}"
        print(msg)
        return []

    segmentations = []

    for _, row in seg_data.iterrows():
        labels = row["labels"]
        multipoly = transform(_flip, row["polygons"])
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


def _get_metadata(id: str) -> ImageMetadata:
    uuid = UUID(id)
    source = _get_source(uuid)

    match = con.table(source).image == uuid
    metadata = con.table(source).filter(match).to_pandas()
    match = con.table("images").uuid == uuid
    imgdata = con.table("images").filter(match).to_pandas()

    if len(metadata) < 1:
        msg = f"No entry found with ID {id}"
        print(msg)
        raise ValueError(msg)
    if len(metadata) > 1:
        msg = f"More than one entry found with ID {id}. Database corrupted?"
        print(msg)
        raise ValueError(msg)
    metadata = metadata.squeeze()
    imgdata = imgdata.squeeze()

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
        source=source,
        tags=imgdata["tags"],
        rating=imgdata["rating"],
        compass_angle=float(metadata["compass_angle"]),
        notes="",
        segmentation=segmentations,
    )


def _inbounds(img: Image | ImageMetadata, bbox: Bbox) -> bool:
    # Temporary implementation.
    # Full implementation needs to account for spherical coordinates properly
    return bbox.n > img.lat > bbox.s and bbox.w > img.lon > bbox.e


def _bbox_to_polygon(bbox: Bbox) -> Polygon:
    return Polygon([(bbox.w,bbox.n),(bbox.e,bbox.n),(bbox.e,bbox.s),(bbox.w,bbox.s)])


def _fetch_images(filter: Optional[FilterParams]) -> list[Image]:
    """Fetch images that conform to a filter specification."""
    if filter is None:
        return _get_images(con.table("mapillary"))

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
    if any((filter.models, filter.labels, filter.model_runs, filter.segmentation_ratings)):
        segmentations = con.table("segmentations")
        # optionally filter for models
        if len(filter.models) > 0:
            runs = con.table("runs")
            runs = runs.filter(runs.model.isin(filter.models))
            segmentations = segmentations.filter(segmentations.run.isin(runs.run))
        
        for label in filter.labels:
            print(f"filtering for label {label}")
            segmentations = segmentations.filter(segmentations.labels.contains(label))

        if len(filter.model_runs) > 0:
            segmentations = segmentations.filter(segmentations.run.isin(filter.model_runs))

        if len(filter.segmentation_ratings) > 0:
            segmentations = segmentations.filter(segmentations.run.isin(filter.segmentation_ratings))

        valid_images = set(segmentations.image.to_pandas())

        _match = mapillary.image.isin(valid_images)
        mapillary = mapillary.filter(_match)
    
    # Now we can request the valid images
    return _get_images(mapillary)


def _unknown_image(image_id, err: Optional[Exception]):
    msg = f"No image found with id '{image_id}'"
    print(msg)
    if err is not None:
        raise HTTPException(status_code=404, detail=msg) from err
    raise HTTPException(status_code=404, detail=msg)


def _get_unique_tags():
    tags = set(chain.from_iterable(con.table("images")["tags"].to_pandas().to_list()))
    return list(tags)


def _get_unique_labels():
    labels = set(chain.from_iterable(con.table("segmentations").labels.to_pandas().to_list()))
    return list(labels)


def _get_daterange() -> tuple[datetime, datetime]:
    # Note: only implemented for mapillary
    mapillary = con.table("mapillary")
    start = datetime.fromtimestamp(mapillary.captured_at.min().to_pandas()/1000)
    end = datetime.fromtimestamp(mapillary.captured_at.max().to_pandas()/1000)
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
async def fetch_stats(bbox: Annotated[Bbox, Query()]) -> AggregateStats:
    """Get the aggregate stats of the images."""
    return AggregateStats(
        tags=_get_unique_tags(),
        labels=_get_unique_labels(),
        model_run_names=list(set(con.table("runs").run.to_pandas())),
        image_sources=list(set(con.table("images")["source"].to_pandas().to_list())),
        date_range=_get_daterange(),
        models=list(set(con.table("runs").model.to_pandas())),
    )


@app.get("/images")
async def fetch_images(filter: Annotated[FilterParams, Query()]) -> list[Image]:
    """Fetch streetscape images corresponding to a bounding box and optionally filters."""
    # bbox = Bbox(**filter.model_dump())
    return _fetch_images(filter)


@app.get("/images/{image_id}")
async def fetch_image_metadata(image_id: str) -> ImageMetadata:
    """Get all metadata associated with a certain image, including segmentations."""
    try:
        return _get_metadata(image_id)
    except ValueError as err:
        _unknown_image(image_id, err)


@app.post("/images/{image_id}/rating")
async def set_rating(image_id: str, rating: int | None):
    """Set an image's rating."""
    for img in _images:
        if img.id == image_id:
            img.rating = rating
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/tags")
async def set_tags(image_id: str, tags: list[str]):
    """Set an image's tags."""
    for img in _images:
        if img.id == image_id:
            img.tags = tags
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/notes")
async def set_notes(image_id: str, notes: str):
    """Set an image's notes."""
    for img in _images:
        if img.id == image_id:
            img.notes = notes
            return None
    _unknown_image(image_id)


@app.post("/images/{image_id}/{segmentation_id}/{instance_idx}/{label}")
async def set_instance_label(
    image_id: str, segmentation_id: str, instance_idx: int, label: str
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
        f"paste in https://localhost:{port} as web service.\n"
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
