from cyclopts import App
from ray import serve
from ray.serve.handle import DeploymentHandle


segment_images_cli = App(help="Segment images")


def _spawn_model_server(model: str) -> DeploymentHandle:
    from streetscapes.serve import model_server
    app = model_server(model)
    print("Starting model...")
    return serve.run(app)


@segment_images_cli.command(name="maskformer")
def segment_images_maskformer(
    image_path: str,  # list[str] | str ; typer doesn't support union
    labels: dict = None,
    batch_size: int = 10,
):
    from streetscapes.models.maskformer.schema import MaskFormerResponseSchema

    if labels is None:
        labels = {"building": None, "sky": None}

    handle = _spawn_model_server("maskformer")

    data = {
        "image_path": image_path,
        "labels": labels,
        "batch_size": batch_size,
    }
    response = handle.remote(data).result()

    if len(response) == 0:
        print(f"==[ Zero-length response :(")

    print(f"==[ Instances: {response[0].instances}")
