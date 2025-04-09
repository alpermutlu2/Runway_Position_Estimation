
import numpy as np
from scipy.optimize import least_squares

def reprojection_residuals(poses, landmarks, observations):
    residuals = []
    for (i, j), obs in observations.items():
        P = poses[i]
        X = landmarks[j]
        projected = P @ np.append(X, 1.0)
        projected /= projected[2]
        residuals.append(obs - projected[:2])
    return np.concatenate(residuals)

def run_bundle_adjustment(initial_poses, initial_landmarks, observations):
    poses_flat = np.concatenate([pose.flatten() for pose in initial_poses])
    landmarks_flat = np.concatenate(initial_landmarks)
    def cost_function(x):
        num_poses = len(initial_poses)
        pose_dim = initial_poses[0].size
        landmark_dim = 3
        poses = [x[i*pose_dim:(i+1)*pose_dim].reshape(3, 4) for i in range(num_poses)]
        landmarks = [x[num_poses*pose_dim + i*landmark_dim:num_poses*pose_dim + (i+1)*landmark_dim] for i in range(len(initial_landmarks))]
        return reprojection_residuals(poses, landmarks, observations)
    result = least_squares(cost_function, np.concatenate([poses_flat, landmarks_flat]))
    return result
