from pathlib import Path

import typer
from fastapi import FastAPI
from ray import serve
from rich.console import Console

from streetscapes.models.maskformer.schema import MaskFormerRequestSchema
from streetscapes.models.maskformer import MaskFormer

app = FastAPI()


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        # TODO: These should be taken from the configuration.
        "num_cpus": 0.2,
        "num_gpus": 0,
    },
)
@serve.ingress(app)
class MaskFormerServer:
    def __init__(self):
        self.model = MaskFormer()
        self.console = Console()

    def _segment(
        self,
        img_path: str,
        labels: dict,
        batch_size: int = 10,
    ) -> str:
        """Perform the segmentation.

        Args:
            img_path: Image path (a single image or a directory of images).
            labels: Labels to focus on.
            batch_size: Batch size for the segmenter.

        Returns:
            Segmentations.
        """
        img_path = Path(img_path).expanduser().resolve().absolute()
        if not img_path.exists():
            return

        # Segment the images
        segmentation = self.model.segment(img_path, labels)

        return segmentation

    @app.post("/segment")
    def segment(
        self,
        req: MaskFormerRequestSchema,
    ) -> list:
        """Endpoint for the segmentation request.

        TODO: Proper response model.

        Args:
            req: A SegmentationRequestModel.

        Returns:
            The segmentation results.
        """
        typer.echo("==[ Request for segmentation received.")

        data = req.model_dump()
        segmentations = self._segment(**data)
        return segmentations

    @app.get("/ping")
    async def ping(self) -> str:
        """Endpoint for checking if the model is alive.

        Returns:
            'pong'
        """
        return "pong"

        return "pong"


segmentation_app = MaskFormerServer.bind()
