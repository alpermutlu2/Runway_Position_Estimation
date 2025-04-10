
import numpy as np
from tracking.orbslam3_wrapper import ORBSLAM3Wrapper
from tracking.lightglue_tracker import LightGlueTracker

class HybridTracker:
    def __init__(self, config):
        self.orb = ORBSLAM3Wrapper()
        self.lg = LightGlueTracker()
        self.config = config
        self.orb.initialize(config)
        self.lg.initialize(config)
        self.use_lightglue = False
        self.last_quality = 1.0  # Start with good quality

    def track(self, image, timestamp):
        if self.last_quality < self.config.get("orbslam3_min_quality", 0.3):
            self.use_lightglue = True
        else:
            self.use_lightglue = False

        if self.use_lightglue:
            pose, kps, is_kf = self.lg.track(image, timestamp)
        else:
            pose, kps, is_kf = self.orb.track(image, timestamp)

        self.last_quality = self._assess_quality(kps)
        return pose, kps, is_kf

    def _assess_quality(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        return min(1.0, len(keypoints) / 200.0)  # normalize to [0, 1]

    def shutdown(self):
        self.orb.shutdown()
        self.lg.shutdown()
