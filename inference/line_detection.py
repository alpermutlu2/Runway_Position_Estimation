# inference/line_detection.py

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


def detect_runway_lines_ransac(image_bgr, seg_mask):
    """Detect left and right runway lines using RANSAC."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    # Mask out non-runway regions
    masked_edges = cv2.bitwise_and(edges, edges, mask=(seg_mask > 0).astype(np.uint8))

    # Get edge points
    y_idxs, x_idxs = np.nonzero(masked_edges)
    if len(x_idxs) < 50:
        return None, None, masked_edges

    # Split into left and right of center
    center_x = image_bgr.shape[1] // 2
    left_mask = x_idxs < center_x
    right_mask = x_idxs >= center_x

    lines = []
    for mask in [left_mask, right_mask]:
        x, y = x_idxs[mask], y_idxs[mask]
        if len(x) < 20:
            lines.append(None)
            continue
        model = RANSACRegressor(residual_threshold=5.0, min_samples=10)
        model.fit(x.reshape(-1, 1), y)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        lines.append((slope, intercept))

    return lines[0], lines[1], masked_edges
