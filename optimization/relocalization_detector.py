
import numpy as np

class RelocalizationDetector:
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def check_loop(self, current_pose, previous_poses):
        cp = current_pose[:3, 3]
        for past_pose in previous_poses:
            pp = past_pose[:3, 3]
            if np.linalg.norm(cp - pp) < self.threshold:
                return True
        return False
