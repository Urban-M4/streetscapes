# import numpy as np
# import torch
# from groundingdino.util.inference import predict, annotate, load_image, load_model
# from PIL import Image


class GroundingDINO:
    pass

# class GroundingDINO:
#     def __init__(self, config_path: str, weights_path: str):
#         self.model = load_model(config_path, weights_path)

#     @torch.no_grad()  # type: ignore
#     def run(
#         self,
#         image: np.ndarray,
#         caption: str,
#         box_threshold: float,
#         text_threshold: float,
#     ) -> np.ndarray:
#         pil_image = Image.fromarray(image)
#         image_source, img = load_image(pil_image)
#         boxes, logits, phrases = predict(
#             model=self.model,
#             image=img,
#             caption=caption,
#             box_threshold=box_threshold,
#             text_threshold=text_threshold,
#         )
#         annotated = annotate(
#             image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
#         )
#         return annotated  # RGB numpy array
