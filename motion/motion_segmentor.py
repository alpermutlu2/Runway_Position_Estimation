
import numpy as np

class MotionSegmentor:
    def __init__(self, flow_threshold=0.5):
        self.prev_flow = None
        self.flow_threshold = flow_threshold

    def segment(self, flow_map):
        if self.prev_flow is None:
            self.prev_flow = flow_map
            return np.ones(flow_map.shape[:2], dtype=np.uint8)
        delta = np.linalg.norm(flow_map - self.prev_flow, axis=2)
        mask = delta < self.flow_threshold
        self.prev_flow = flow_map
        return mask.astype(np.uint8)
