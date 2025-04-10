
import numpy as np

class KeyframeLogger:
    def __init__(self):
        self.keyframes = []

    def log(self, frame_id, pose, semantics=None):
        self.keyframes.append({
            'frame_id': frame_id,
            'pose': pose,
            'semantics': semantics
        })

    def get_recent(self, N=5):
        return self.keyframes[-N:]

    def get_all_poses(self):
        return [kf['pose'] for kf in self.keyframes]

    def get_semantic_counts(self):
        label_counts = {}
        for kf in self.keyframes:
            if kf['semantics']:
                for label in kf['semantics']:
                    label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
