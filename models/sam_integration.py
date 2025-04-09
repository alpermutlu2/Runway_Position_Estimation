import numpy as np
from segment_anything import SamPredictor, sam_model_registry

class SAMDynamicMasker:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h.pth"):
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.model)
    
    def generate_mask(self, image):
        """Input: np.ndarray (H, W, 3), Output: binary mask (H, W)"""
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict()
        return masks[0]  # Use first mask