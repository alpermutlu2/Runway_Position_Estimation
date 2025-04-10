
import numpy as np
import cv2

class FrameQualityClassifier:
    def __init__(self, entropy_thresh=4.5, kp_thresh=100):
        self.entropy_thresh = entropy_thresh
        self.kp_thresh = kp_thresh

    def compute_entropy(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        return -np.sum([p * np.log2(p) for p in hist if p > 0])

    def decide(self, image, keypoints):
        entropy = self.compute_entropy(image)
        kp_count = len(keypoints)
        use_lightglue = entropy < self.entropy_thresh or kp_count < self.kp_thresh
        return 'lightglue' if use_lightglue else 'orbslam3'
