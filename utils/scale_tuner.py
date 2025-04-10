
import numpy as np

class ScaleTuner:
    def __init__(self, ref_camera_height=1.2):
        self.ref_height = ref_camera_height

    def compute_scale(self, depth_map, pixel_region=None):
        h, w = depth_map.shape
        cx, cy = w // 2, h // 2
        region = depth_map[cy-5:cy+5, cx-5:cx+5] if pixel_region is None else pixel_region
        mean_depth = np.mean(region)
        scale_factor = self.ref_height / (mean_depth + 1e-6)
        return scale_factor
