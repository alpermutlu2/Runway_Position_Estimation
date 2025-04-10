
import numpy as np

class Evaluator:
    def __init__(self):
        self.poses = []
        self.positions = []

    def log(self, frame_id, pose, position):
        self.poses.append(pose)
        self.positions.append(position)

    def summarize(self):
        print("\n=== Evaluation Summary ===")
        print(f"Total Frames: {len(self.positions)}")
        traj = np.array(self.positions)
        drift = np.linalg.norm(traj[-1] - traj[0]) if len(traj) > 1 else 0
        print(f"Estimated Drift: {drift:.3f} meters")
