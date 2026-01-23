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
    model_name="dummy-model",
    id="",
    run_args="no-args",
    instances=[_instance],
    notes="",
)


_images = [
    ImageMetadata(
        id="4938291029384",
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
        id="4938291029385",
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
        segmentation=[_segmentation,],
    ),
    ImageMetadata(
        id="813431763582202",
        url="https://scontent-ams2-1.xx.fbcdn.net/m1/v/t6/An9faJ8rGWxrgJo0LQuSSFfwjCIQs68cdV5sT6Q6JhyekKEMCCeyYIpNE9yfzCvr-Y2-GcsOUvZ-bN2a08b8an9_GgpfqLTbzvyOAQi6A3xIfeedNSKT9qIjz_noK5U417zyn_V1iUvQ9tOsTotfhQ?edm=AOnQwmMEAAAA&_nc_gid=I5IKYZGHOrMc5-0IONvYfA&_nc_oc=AdkagW3_wNIJmGy3-gp_7dpuw-9vpJNXrfGlKxldRLLD3qSPRmh9Oe2sUu_OOwgZxCY&ccb=10-5&oh=00_AfpR8kWxxWvr7eJvNwUTFBW3mBbMAgIaAbbLv8hMZ9QXYQ&oe=699A6D80&_nc_sid=201bca",
        lat=52.3709772,
        lon=4.8902233,
        altitude=57.984,
        captured_at=datetime.fromtimestamp(1694259141),
        source="mapillary",
        panoramic=1,
        compass_angle=194.525,
        width=5760,
        height=2880,
        notes="GoPro Max panorama",
    ),
]


