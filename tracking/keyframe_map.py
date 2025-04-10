
import numpy as np
import cv2

class KeyframeMap:
    def __init__(self):
        self.keyframes = []

    def store(self, image, keypoints, descriptors, pose):
        self.keyframes.append({
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose
        })

    def find_best_match(self, query_desc, matcher):
        best_idx = -1
        best_score = float('inf')
        for i, kf in enumerate(self.keyframes):
            matches = matcher.match(query_desc, kf['descriptors'])
            score = sum([m.distance for m in matches]) / (len(matches) + 1e-6)
            if score < best_score:
                best_score = score
                best_idx = i
        return self.keyframes[best_idx] if best_idx >= 0 else None
