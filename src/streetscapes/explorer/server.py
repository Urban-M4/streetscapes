"""FastAPI server implementation."""

import asyncio
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID

import ibis
import pandas as pd
import uvicorn
from cyclopts import App, Parameter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Query
from shapely.ops import transform

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
    "http://localhost",
    "https://localhost",
    "http://localhost:5173",
    "https://urban-m4.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
dbpath = Path(f"{CFG.project_dir}/{CFG.active_project}.duckdb")
con = ibis.duckdb.connect(dbpath, read_only=True, extensions=["spatial", "json"])


def _get_images():
    data = (
        con.table("mapillary")
        .select("image", "thumb_original_url", "geometry")
        .to_pandas()
    )

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
        seg = Segmentation(
            model_name=runinfo["model"],
            id=row["run"],
            run_args=runinfo["metadata"].encode("utf8").decode("unicode_escape"),
            instances=inst,
        )
        segmentations.append(seg)
    return segmentations


def _get_metadata(id: str) -> ImageMetadata:
    uuid = UUID(id)
    source = _get_source(uuid)

    match = con.table(source).image == uuid
    data = con.table(source).filter(match).to_pandas()

    if len(data) < 1:
        msg = f"No entry found with ID {id}"
        print(msg)
        raise ValueError(msg)
    if len(data) > 1:
        msg = f"More than one entry found with ID {id}. Database corrupted?"
        print(msg)
        raise ValueError(msg)
    data = data.squeeze()

    segmentations = _get_segmentations(uuid)

    return ImageMetadata(
        id=str(data["image"]),
        url=data["thumb_2048_url"],
        lat=data["geometry"].y,
        lon=data["geometry"].x,
        width=int(data["width"]),
        height=int(data["height"]),
        altitude=data["altitude"],
        captured_at=datetime.fromtimestamp(data["captured_at"] / 1000),
        panoramic=bool(data["is_pano"]),
        source=source,
        tags=[],
        rating=0,
        compass_angle=float(data["compass_angle"]),
        notes="",
        segmentation=segmentations,
    )


def _inbounds(img: Image | ImageMetadata, bbox: Bbox) -> bool:
    # Temporary implementation.
    # Full implementation needs to account for spherical coordinates properly
    return bbox.n > img.lat > bbox.s and bbox.w > img.lon > bbox.e


def _fetch_images(bbox: Optional[Bbox]) -> list[Image]:
    if bbox is not None:
        return _get_images()
    return _get_images()


def _unknown_image(image_id, err: Optional[Exception] = None):
    msg = f"No image found with id '{image_id}'"
    print(msg)
    if err is not None:
        raise HTTPException(status_code=404, detail=msg) from err
    raise HTTPException(status_code=404, detail=msg)


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
        tags=["sunny", "shops", "crowded"],
        labels=[
            "vegetation",
            "car",
            "building",
            "bicycle",
            "person",
            "sky",
            "water",
            "terrain",
            "pedestrian-area",
        ],
        model_run_names=["manual"],
        image_sources=[
            "mapillary",
        ],
        date_range=(datetime(2026, 1, 19), datetime(2026, 1, 20)),
        models=["DinoSAM", "maskformer", "bfms", "manual"],
    )


@app.get("/images")
async def fetch_images(filter: Annotated[FilterParams, Query()]) -> list[Image]:
    """Fetch streetscape images corresponding to a bounding box and optionally filters."""
    # bbox = Bbox(**filter.model_dump())
    return _fetch_images(None)


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


async def _start_uvicorn(port: int, log_info: bool):
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info" if log_info else "warning",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _serve(port: int, open_webpage: bool, log_info: bool):
    server = _start_uvicorn(port, log_info)
    if open_webpage:
        print("Waiting for the streetscapes-explorer to start...")
        await asyncio.sleep(5)
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
        f"paste in https://0.0.0.0:{port} as web service.\n"
        "  You will need to disable your ad blocker (like uBlock Origin Lite)\n"
        "and allow your web browser to load localhost resources."
    )
    await server


cli = App(help="Streetscapes data explorer")

@cli.default
async def serve(
    *,
    port: Annotated[int, Parameter(name=["--port", "-p"])] = 5001,
    open_webpage: bool = True,
    verbose_logs: bool = False,
):
    """Start the Streetscapes Explorer server.
    
    Args:
        port: port to host the backend on.
        open_webpage: automatically open a browser window with the frontend viewer,
            with the backend correctly configured.
        verbose_logs: display verbose backend server logs, useful for debugging the
            frontend.
    """
    await _serve(port, open_webpage, verbose_logs)

if __name__ == "__main__":
   cli()
