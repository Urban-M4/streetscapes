import os
from typing import Any

from ray import serve
from ray.serve.handle import DeploymentHandle

from rich.console import Console

from streetscapes.models.bfms import BFMS
from streetscapes.models.dinosam import DinoSAM
from streetscapes.models.maskformer import MaskFormer

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        # TODO: These should be taken from the configuration.
        "num_cpus": 0.2,
        "num_gpus": 0,
    },
)
class ModelApp:

    available_models = {
        m.__name__.lower(): m
        for m in (
            MaskFormer,
            DinoSAM,
            BFMS,
        )
    }

    def __init__(self, model: str, /, **kwargs):

        self.con = Console()
        self.con.print(f"Starting model: {model}")

        model = model.lower()

        if model not in self.available_models:
            raise KeyError(f"Invalid model '{model}'")

        self.model = self.available_models[model](**kwargs)

    async def __call__(self, request: Any):

        self.con.print(f"{self.model.name} model received a segmentation request.")
        return await self.model.process(request)


def get_model_app(model: str, /, **kwargs) -> serve.Application:
    return ModelApp.bind(model, **kwargs)


def serve_model(model: str, /, **kwargs) -> DeploymentHandle:

    app = get_model_app(model, **kwargs)
    return serve.run(app)
