
import numpy as np

class PoseFilter:
    def __init__(self, alpha=0.8, beta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.filtered_position = None
        self.last_position = None

    def filter(self, position):
        if self.filtered_position is None:
            self.filtered_position = position
            self.last_position = position
        else:
            # Blend with current position (low-pass + EMA)
            delta = position - self.last_position
            self.last_position = position
            self.filtered_position = (
                self.alpha * self.filtered_position + self.beta * delta + (1 - self.alpha - self.beta) * position
            )
        return self.filtered_position
