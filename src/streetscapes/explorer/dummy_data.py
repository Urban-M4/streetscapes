from datetime import datetime
from streetscapes.explorer.data import ImageMetadata, Instance, Segmentation


_instance = Instance(
    label="triangles",
    polygon=[
        [(0,0),(155,235),(0,300),(0,0)],
        [(50,800),(155,235),(0,900),(50,800)],
    ],
)
_segmentation = Segmentation(
    "dummy-model",
    "no-args",
    (_instance),
    notes="",
)
_images = [
    ImageMetadata(
        id=0,
        url=r"https://upload.wikimedia.org/wikipedia/commons/0/00/Zaden_van_een_Gele_lis_%28Iris_pseudacorus%29._06-03-2024._%28d.j.b.%29.jpg",
        lat=52.3751914,
        lon=4.8954506,
        altitude=0.0,
        captured_at=datetime(2026, 1, 20),
        source="wikimedia-commons",
        width=3454,
        height=5182,
        notes="Gele Iris",
    ),
    ImageMetadata(
        id=1,
        url=r"https://upload.wikimedia.org/wikipedia/commons/b/b8/Chestnut-naped_antpitta_%28Grallaria_nuchalis_ruficeps%29_Las_Tangaras.jpg",
        lat=52.3727217,
        lon=4.9003963,
        altitude=0.0,
        captured_at=datetime(2026, 1, 19),
        source="wikimedia-commons",
        tags=[
            "birb",
        ],
        width=3092,
        height=4000,
        notes="is cute",
        segmentation=(_segmentation,),
    ),
]