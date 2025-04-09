
import numpy as np

def evaluate_trajectory(predicted, ground_truth):
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    translation_error = np.linalg.norm(predicted - ground_truth, axis=1)
    ate = np.mean(translation_error)
    diffs = predicted[1:] - predicted[:-1]
    gt_diffs = ground_truth[1:] - ground_truth[:-1]
    rpe = np.mean(np.linalg.norm(diffs - gt_diffs, axis=1))
    return {'ATE': ate, 'RPE': rpe}
