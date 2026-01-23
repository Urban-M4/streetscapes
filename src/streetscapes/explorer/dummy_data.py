from datetime import datetime
from streetscapes.explorer.data import ImageMetadata, Instance, Segmentation

_images = [
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
        tags=["sunny"],
        segmentation=[
            Segmentation(
                model_name="manual",
                id="manual",
                run_args="",
                instances=[
                    Instance(
                        label="building",
                        polygon=[
                            [
                                (2213.632080078125, 1245.4053955078125),
                                (1674.4627685546875, 1442.8477783203125),
                                (1674.4627685546875, 1442.8477783203125),
                                (2001.001953125, 1716.2293701171875),
                                (2266.789794921875, 1556.7567138671875),
                            ]
                        ],
                    ),
                    Instance(
                        label="car",
                        polygon=[
                            [
                                (3413.473876953125, 1792.168701171875),
                                (3497.00732421875, 1473.2235107421875),
                                (3497.00732421875, 1473.2235107421875),
                                (4051.364501953125, 1670.665771484375),
                                (3975.42529296875, 1868.108154296875),
                            ]
                        ],
                    ),
                ],
            ),
            Segmentation(
                model_name="sam3",
                id="sam3-run-001",
                run_args="",
                instances=[
                    Instance(
                        label="tree",
                        polygon=[
                            [
                                (656.8754272460938, 1450.441650390625),
                                (1271.984130859375, 1625.1021728515625),
                                (1271.984130859375, 1625.1021728515625),
                                (1180.85693359375, 1890.889892578125),
                                (634.0936279296875, 1723.8233642578125),
                            ]
                        ],
                    ),
                ],
                notes="",
            ),
        ],
    ),
    ImageMetadata(
        id="465548397995389",
        url="https://scontent-ams2-1.xx.fbcdn.net/m1/v/t6/An-aPrM8udPtbeQxdkLlx3Z11CXXAnc4j0mbj6COklNObHH5GhQqOaU7iRKqxK7JqGPcERRu83ri5wpuZplNsx-8ZTqfqA4mjkfuzTQdaRFrNMC79WjW1EN52QMT16A1TkaS1Vkf2PEdOCTL0h6tvA?edm=AOnQwmMEAAAA&_nc_gid=I5IKYZGHOrMc5-0IONvYfA&_nc_oc=AdmNhT0KwfQARoMOviXmvLDsV92KnOrLXj2MCGlSAzmdqGZq1PikeyYMmx5dS0Xyc1A&ccb=10-5&oh=00_AfraeLaz6MK5ZVN_-_dXYAjgp_91LofegluWUGYdjo7Qyw&oe=699A82C5&_nc_sid=201bca",
        lat=52.370137729191,
        lon=4.8995452556366,
        altitude=49.217982324772,
        captured_at=datetime.fromtimestamp(1465460016.497),
        source="amsterdam",
        width=8000,
        height=4000,
        panoramic=1,
        compass_angle=0.0,
        notes="Trimble TMX pano, sequence XjKEty58SjjIwD6A6T_42A",
        tags=["sunny"],
    ),
    ImageMetadata(
        id="3011051359053270",
        url="https://scontent-ams2-1.xx.fbcdn.net/m1/v/t6/An-ZtBzLAk44LSmUSQIgQjjdDOdGrzGXOhm-1jkgNWtqzygqTmxanWAOzUuLINGBJXQAl4St5pPJeQ-GCBIW9QvJRUYhZDNLgzjDjEYzPTLHpkP-W9I8yJKbk27KG3p1xqkS02vzfQVAOhyIrqy6ypU?edm=AOnQwmMEAAAA&_nc_gid=I5IKYZGHOrMc5-0IONvYfA&_nc_oc=Adn1wezvRavRT7iKXh1VFWD_nkR0tPtNseQgDa9bEwf7J25xEEcgjDuwvq93IVwBTi4&ccb=10-5&oh=00_Afr46Q75ENE9t5xVPM3AAVH1YMxil8grKUkgkELAuCSBRQ&oe=699A7198&_nc_sid=201bca",
        lat=52.371349171381,
        lon=4.8920353200333,
        altitude=68.182579049736,
        captured_at=datetime.fromtimestamp(1746292329.2),
        source="mapillary",
        width=2704,
        height=2028,
        notes="GoPro HERO7 Black",
        compass_angle=336.85915920365,
        panoramic=False,
        tags=["shops", "crowded"],
        segmentation=[
            Segmentation(
                model_name="manual",
                id="manual",
                run_args="",
                instances=[
                    Instance(
                        label="building",
                        polygon=[
                            [
                                (3.7969677448272705, 403.9024353027344),
                                (489.808837890625, 2598.5498046875),
                                (489.808837890625, 2598.5498046875),
                                (1294.7659912109375, 2112.537841796875),
                                (1385.8931884765625, 715.2537841796875),
                                (1803.5596923828125, 692.4719848632812),
                                (1826.3414306640625, 874.7264404296875),
                                (2570.547119140625, 1087.3565673828125),
                                (2760.3955078125, 1079.7626953125),
                                (2813.552978515625, 745.6295166015625),
                                (2957.837890625, -135.2669677734375),
                                (2813.552978515625, -682.0303344726562),
                                (3.7969677448272705, -689.624267578125),
                            ]
                        ],
                    ),
                ],
            ),
        ],
    ),
]
