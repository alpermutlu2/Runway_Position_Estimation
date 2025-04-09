
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

class SAMDynamicMasker:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h.pth", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

    def generate_mask(self, image, threshold=0.5):
        """
        Input: image - np.ndarray of shape (H, W, 3), dtype uint8 or float32 [0-1 or 0-255]
        Output: binary mask of shape (H, W), dtype bool
        """
        assert isinstance(image, np.ndarray), "Input must be a numpy array"
        assert image.ndim == 3 and image.shape[2] == 3, "Input must be (H, W, 3)"
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict()

        if masks is None or len(masks) == 0:
            return np.zeros(image.shape[:2], dtype=bool)

        # Option 1: Combine all masks above threshold
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for m, s in zip(masks, scores):
            if s >= threshold:
                combined_mask |= m.astype(bool)

        return combined_mask
