
class KeyframeManager:
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.keyframes = []

    def should_add_keyframe(self, current_pose, last_keyframe_pose):
        if len(self.keyframes) == 0:
            return True
        movement = ((current_pose[:3, 3] - last_keyframe_pose[:3, 3])**2).sum()**0.5
        return movement > self.threshold

    def add_keyframe(self, pose, frame_id):
        self.keyframes.append((frame_id, pose))
