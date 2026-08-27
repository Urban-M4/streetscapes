"""CV model serving."""

import logging
import os
from typing import TYPE_CHECKING, Any

import ray
from ray import serve
from rich.console import Console  # TODO: import from cli.console, or just use logger?

from streetscapes.models.bfms.service import BFMSService
from streetscapes.models.dinosam.service import DinoSAMService
from streetscapes.models.maskformer.service import MaskFormerService

if TYPE_CHECKING:
    from ray.serve.handle import DeploymentHandle


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
    """CV model app wrapper."""

    def __init__(self, model: str, /, **kwargs):
        """Initialize a model."""
        self.con = Console()
        self.con.print(f"Starting model: {model}")

        model = model.lower()

        if model not in MODEL_REGISTRY:
            raise KeyError(f"Invalid model '{model}'")

        self.model = model
        self.service = MODEL_REGISTRY[model](**kwargs)

    async def __call__(self, request: Any):
        """Make a model request."""
        self.con.print(f"Processing request for model '{self.model}'.")
        return self.service.handle(request)


def get_model_app(model: str, /, **kwargs) -> serve.Application:
    """Get model application in preparation of serving."""
    return ModelApp.bind(model, **kwargs)  # type: ignore[attr-defined,no-any-return]


def serve_model(
    model: str, verbose: bool = False, /, **model_kwargs
) -> DeploymentHandle:
    """Serve CV model using ray."""
    app = get_model_app(model, **model_kwargs)

    logger = logging.getLogger("ray.serve")
    if not verbose:
        ray.init(log_to_driver=False)
        logger.setLevel(logging.WARNING)
    return serve.run(  # type: ignore[no-any-return]
        app, logging_config={"log_level": logging.INFO if verbose else logging.WARNING}
    )
