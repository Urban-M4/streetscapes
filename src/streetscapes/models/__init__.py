from streetscapes.models.base import ImagePath
from streetscapes.models.base import BaseSegmenter
from streetscapes.models.maskformer import MaskFormer
from streetscapes.models.dinosam import DinoSAM
from streetscapes.models.base import load_model

# Register available models
BaseSegmenter._register_models()
