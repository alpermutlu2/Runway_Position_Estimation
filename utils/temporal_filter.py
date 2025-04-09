
import numpy as np

class TemporalFilter:
    def __init__(self, method='ema', alpha=0.9):
        self.method = method
        self.alpha = alpha
        self.prev = None

    def apply(self, data_sequence):
        smoothed = []
        for frame in data_sequence:
            if self.prev is None:
                smoothed.append(frame)
            else:
                smoothed_frame = self.alpha * frame + (1 - self.alpha) * self.prev
                smoothed.append(smoothed_frame)
                self.prev = smoothed_frame
            self.prev = frame
        return smoothed
