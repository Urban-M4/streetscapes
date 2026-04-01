import logging
import os
from typing import Any

from ray import serve
from ray.serve.handle import DeploymentHandle
from rich.console import Console  # TODO: import from cli.console, or just use logger?
import ray

from streetscapes.models.bfms.service import BFMSService
from streetscapes.models.maskformer.service import MaskFormerService
from streetscapes.models.dinosam.service import DinoSAMService

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

MODEL_REGISTRY = {
    "bfms": BFMSService,
    "maskformer": MaskFormerService,
    "dinosam": DinoSAMService,
}


@serve.deployment(
    num_replicas=1,
)
class ModelApp:
    def __init__(self, model: str, /, **kwargs):

        self.con = Console()
        self.con.print(f"Starting model: {model}")

        model = model.lower()

        if model not in MODEL_REGISTRY:
            raise KeyError(f"Invalid model '{model}'")

        self.model = model
        self.service = MODEL_REGISTRY[model](**kwargs)

    async def __call__(self, request: Any):
        self.con.print(f"Processing request for model '{self.model}'.")
        return self.service.handle(request)


def get_model_app(model: str, /, **kwargs) -> serve.Application:
    return ModelApp.bind(model, **kwargs)  # type: ignore[attr-defined,no-any-return]


def serve_model(model: str, verbose: bool = False, /, **model_kwargs) -> DeploymentHandle:
    app = get_model_app(model, **model_kwargs)

    logger = logging.getLogger("ray.serve")
    if not verbose:
        ray.init(log_to_driver=False)
        logger.setLevel(logging.WARNING)
    return serve.run(
        app, logging_config={"log_level": logging.INFO if verbose else logging.WARNING}
    )
