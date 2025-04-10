
from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def initialize(self, config):
        pass

    @abstractmethod
    def track(self, image, timestamp):
        """Returns: pose (4x4), keypoints, is_keyframe (bool)"""
        pass

    @abstractmethod
    def shutdown(self):
        pass
