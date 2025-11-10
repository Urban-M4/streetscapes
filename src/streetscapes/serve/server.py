import os
from typing import Any
from ray import serve
from rich.console import Console

from streetscapes.models.maskformer import MaskFormer
from streetscapes.models.dinosam import DinoSAM
from streetscapes.models.bfms import BFMS

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        # TODO: These should be taken from the configuration.
        "num_cpus": 0.2,
        "num_gpus": 0,
    },
)
class ModelServer:

    available_models = {
        m.__name__.lower(): m
        for m in (
            MaskFormer,
            DinoSAM,
            BFMS,
        )
    }

    def __init__(self, model: str, *args, **kwargs):

        self.con = Console()
        self.con.print(f"Starting model: {model}")

        if model not in self.available_models:
            raise KeyError(f"Invalid model '{model}'")

        self.model = self.available_models[model]()

    async def __call__(self, request: Any):

        self.con.print(f"{self.model.name} model received a segmentation request.")
        return await self.model.process(request)


def model_server(model: str, *args, **kwargs) -> serve.Application:
    return ModelServer.bind(model, *args, **kwargs)
