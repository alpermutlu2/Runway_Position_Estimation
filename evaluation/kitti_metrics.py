
import numpy as np

def load_kitti_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def compute_ATE(gt_poses, est_poses):
    gt = np.array([p[:3, 3] for p in gt_poses])
    est = np.array([p[:3, 3] for p in est_poses])
    length = min(len(gt), len(est))
    gt, est = gt[:length], est[:length]
    ate = np.linalg.norm(gt - est, axis=1).mean()
    return ate
