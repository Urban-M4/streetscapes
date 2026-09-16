"""SAM3 predictor with memory-efficient mask post-processing."""

import torch
import torch.nn.functional as F
import torchvision
from ultralytics.engine.results import Results
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils import ops

CHUNK = 8


class LowMemorySAM3SemanticPredictor(SAM3SemanticPredictor):
    """SAM3 semantic predictor that upscales the predicted masks in small chunks.

    Ultralytics upscales all kept masks to the original image size as a single
    float32 tensor before thresholding them, which costs 4 bytes per pixel per
    instance (~4 GB for 80 instances on a 12 MP photo). Each mask is treated
    independently, so upscaling and thresholding them a couple at a
    time yields identical masks while consuming much less memory.
    Benchmarking found chunks of 8 to be about as fast as Ultralyics orignal
    single-batch upscaling while avoiding possible OOMs on large images
    with many instances.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Filter the predictions and build the results.

        Mirrors `SAM3SemanticPredictor.postprocess` from ultralytics 8.4, except for
        the mask upscaling.
        """
        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat(
            [pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1
        )

        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = pred_masks[keep], pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)  # classes
        nms_boxes = pred_boxes[:, :4] + c  # boxes (offset by class)
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)
        pred_boxes, pred_masks = pred_boxes[keep], pred_masks[keep]

        names = getattr(
            self.model, "names", [str(i) for i in range(pred_scores.shape[0])]
        )
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip(
            [pred_masks], [pred_boxes], orig_imgs, self.batch[0]
        ):
            if masks.shape[0] == 0:
                masks, boxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = self._upscale_masks(masks, orig_img.shape[:2])
                boxes[..., [0, 2]] *= orig_img.shape[1]
                boxes[..., [1, 3]] *= orig_img.shape[0]
            results.append(
                Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes)
            )
        return results

    def _upscale_masks(
        self, masks: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """Upscale (N, h, w) logits to boolean (N, *size) masks, `CHUNK` at a time."""
        upscaled = torch.empty(
            (masks.shape[0], *size), dtype=torch.bool, device=masks.device
        )
        for start in range(0, masks.shape[0], CHUNK):
            torch.gt(
                F.interpolate(
                    masks[start : start + CHUNK].float()[None], size, mode="bilinear"
                )[0],
                self.model.mask_threshold,  # type: ignore[attr-defined]
                out=upscaled[start : start + CHUNK],
            )
        return upscaled
