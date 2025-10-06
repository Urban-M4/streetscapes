import numpy as np
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class CLIPSeg:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray, prompt: str):
        pil_img = Image.fromarray(image)
        inputs = self.processor(text=[prompt], images=[pil_img], return_tensors="pt")
        logits = self.model(**inputs).logits[0, 0]
        mask = torch.sigmoid(logits).cpu().numpy()
        return mask
