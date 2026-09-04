import uuid
from datetime import UTC

import ibis
import shapely as shp
from cyclopts import App
from mimesis import Field, Fieldset, Schema, keys
from mimesis.locales import Locale
from mimesis.schema import SchemaBuilder
from pandas import DataFrame

from streetscapes import utils
from streetscapes.project import Project

bootstrap_cli = App(help="Bootstrap project database.")
ibis.options.interactive = True


def _populate_builder(
    field: Field,
    fieldset: Fieldset,
    builder: SchemaBuilder,
) -> dict:
    """
    Populate the builder with schema

    Args:
        field: A Field generator.
        fieldset: A Fieldset generator.
        builder: A SchemaBuilder generator.

    Returns:
        Generated data.
    """

    source_choices = {
        "mapillary": 1.0,
        "kartaview": 0.0,
        "amsterdam": 0.0,
    }

    model_choices = {
        "maskformer": 0.6,
        "dinosam": 0.2,
        "bfms": 0.2,
    }

    # Runs
    builder.define(
        "runs",
        Schema(
            lambda: {
                "run": utils.uuid7(),
                "timestamp": utils.iso_timestamp("microseconds"),
                "model": field.get_random_instance().weighted_choice(
                    choices=model_choices
                ),
                "metadata": {
                    "key": "value",
                    "custom": field("sentence"),
                    "none": None,
                },
            }
        ),
    )

    # Segmentations
    builder.define(
        "segmentations",
        Schema(
            lambda: {
                "curated": field("development.boolean"),
                "labels": fieldset("text.word", i=3),
                "rating": field("numeric.integer_number", start=0, end=5),
                "confidences": fieldset(
                    "numeric.float_number", start=0.0, end=1.0, i=3
                ),
                "polygons": shp.GeometryCollection(
                    [
                        shp.MultiPolygon(
                            [shp.Polygon(field("numeric.matrix", m=10, n=2))]
                        )
                        for _ in range(3)
                    ]
                ),
            }
        ).map(
            lambda item, ctx: {
                "run": ctx.pick_from("runs", "run"),
                "curated": item["curated"],
                "image": ctx.pick_from("images", "uuid"),
                "labels": item["labels"],
                "rating": item["rating"],
                "polygons": item["polygons"],
                "confidences": item["confidences"],
            }
        ),
    )

    # Mapillary
    builder.define(
        "mapillary",
        Schema(
            lambda: {
                "image": uuid.uuid4(),
                "altitude": field("numeric.float_number", start=0.0, end=10000.0),
                "atomic_scale": field("numeric.float_number", start=0.0, end=1.0),
                "camera_type": field("text.word"),
                "captured_at": field(
                    "datetime.datetime",
                    start=2015,
                    end=2025,
                    key=lambda dt: dt.replace(tzinfo=UTC),
                ),
                "compass_angle": field("numeric.float_number", start=0.0, end=360.0),
                "computed_altitude": field(
                    "numeric.float_number", start=0.0, end=10000.0
                ),
                "computed_compass_angle": field(
                    "numeric.float_number", start=0.0, end=1.0
                ),
                "computed_geometry": shp.Polygon(field("numeric.matrix", m=15, n=2)),
                "computed_rotation": fieldset(
                    "numeric.float_number", start=0.0, end=360.0
                ),
                "creator": {
                    "first_name": field("person.first_name"),
                    "last_name": field("person.last_name"),
                },
                "exif_orientation": field("numeric.integer_number", start=0, end=100),
                "geometry": shp.Polygon(field("numeric.matrix", m=15, n=2)),
                "height": field("numeric.integer_number", start=0, end=100),
                "id": field("numeric.integer_number", start=0, end=100),
                "is_pano": field("development.boolean"),
                "make": field("text.word"),
                "model": field("text.word"),
                "sequence": field("text.word"),
                "thumb_1024_url": field("url"),
                "thumb_2048_url": field("url"),
                "thumb_256_url": field("url"),
                "thumb_original_url": field("url"),
                "width": field("numeric.integer_number", start=0, end=2048),
                "camera_parameters": fieldset(
                    "numeric.float_number", start=0.0, end=100.0, i=3
                ),
            }
        ),
    )

    # Define the users schema
    builder.define(
        "images",
        Schema(
            lambda: {
                "uuid": uuid.UUID(field("uuid")),
                "source": field.get_random_instance().weighted_choice(
                    choices=source_choices
                ),
                "shard": field(
                    "shard",
                    parts_count=3,
                    key=lambda p: "/".join([k[:2] for k in p.split("/")]),
                ),
                "notes": field("sentence"),
                "tags": field(
                    "choice",
                    items=[f"tag{i}" for i in range(10)],
                    length=5,
                    key=lambda x: list(set(x)),
                ),
                "rating": field("numeric.integer_number", start=0, end=5),
            }
        ).map(
            lambda item, ctx: {
                "uuid": ctx.pick_from("mapillary", "image"),
                **item,
            }
        ),
    )


def bootstrap(
    project: str | None = None,
    images: int = 30,
    mapillary: int = 30,
    segmentations: int = 100,
    runs: int = 30,
):
    """
    Bootstrap the database.

    Args:
        project: Project name.
        images: Number of images.
        mapillary: Number of Mapillary entries.
        segmentations: Number of segmentations.
        runs: Number of runs.
    """

    # Mimesis objects
    SEED = None
    field = Field(Locale.EN, seed=SEED)
    fieldset = Fieldset(Locale.EN, seed=SEED)
    builder = SchemaBuilder(seed=SEED)

    # Random project name
    if project is None:
        project = field("text.word", key=keys.prefix("temp_project_"))

    proj = Project(project)
    proj.bootstrap(overwrite=True)

    _populate_builder(field, fieldset, builder)

    generated = builder.create(
        mapillary=mapillary,
        images=images,
        runs=runs,
        segmentations=segmentations,
    )

    for t in generated:
        df = DataFrame(generated[t])
        proj._con.con.register("fake_data", df)
        proj._con.raw_sql(f"INSERT OR IGNORE INTO {t} FROM fake_data;")
        proj._con.con.unregister("fake_data")

    print(proj.table("runs").head())


bootstrap_cli.command(bootstrap, name="bootstrap")

if __name__ == "__main__":
    bootstrap_cli()
