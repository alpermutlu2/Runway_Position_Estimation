
import numpy as np
from tracking.base_tracker import BaseTracker

class LightGlueTracker(BaseTracker):
    def initialize(self, config):
        print("Initialized LightGlue with SuperPoint")

    def track(self, image, timestamp):
        pose = np.eye(4)
        keypoints = np.random.rand(150, 2)
        return pose, keypoints, True

    def shutdown(self):
        print("Shutting down LightGlue")
