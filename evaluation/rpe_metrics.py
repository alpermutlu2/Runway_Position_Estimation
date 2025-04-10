
import numpy as np

def compute_RPE(gt_poses, est_poses, delta=1):
    trans_errors = []
    len_poses = min(len(gt_poses), len(est_poses))
    for i in range(len_poses - delta):
        gt1, gt2 = gt_poses[i], gt_poses[i + delta]
        est1, est2 = est_poses[i], est_poses[i + delta]

        gt_rel = np.linalg.inv(gt1) @ gt2
        est_rel = np.linalg.inv(est1) @ est2

        error = np.linalg.inv(gt_rel) @ est_rel
        trans_error = np.linalg.norm(error[:3, 3])
        trans_errors.append(trans_error)

    return np.mean(trans_errors), np.std(trans_errors)
