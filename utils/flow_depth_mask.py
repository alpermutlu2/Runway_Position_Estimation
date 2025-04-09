
import numpy as np
import cv2

def compute_flow_depth_mask(flow, depth1, depth2, threshold=0.1):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    warped_x = (grid_x + flow[..., 0]).astype(np.float32)
    warped_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped_depth2 = cv2.remap(depth2, warped_x, warped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    depth_diff = np.abs(warped_depth2 - depth1)
    mask = (depth_diff < threshold).astype(np.uint8)
    return mask
