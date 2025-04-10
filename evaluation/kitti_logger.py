
import numpy as np
import os

class KITTILogger:
    def __init__(self, save_path="poses.txt"):
        self.save_path = save_path
        self.poses = []

    def log(self, pose):
        T = pose[:3, :4].reshape(-1)  # 3x4 matrix as 12-element row
        self.poses.append(T)

    def save(self):
        with open(self.save_path, 'w') as f:
            for T in self.poses:
                f.write(' '.join(f"{x:.6f}" for x in T) + '\n')
        print(f"Saved estimated poses to {self.save_path}")
