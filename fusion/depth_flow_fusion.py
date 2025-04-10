
import numpy as np

class DepthFlowFusion:
    def __init__(self):
        print("Fusion module with uncertainty monitoring.")

    def fuse(self, depth_map, flow_map, mask=None):
        flow_mag = np.linalg.norm(flow_map, axis=2)
        depth_conf = 1.0 / (1.0 + np.abs(depth_map - np.median(depth_map)))
        flow_conf = flow_mag / (np.max(flow_mag) + 1e-6)

        # Normalize confidences
        depth_conf = (depth_conf - depth_conf.min()) / (depth_conf.max() - depth_conf.min() + 1e-6)
        flow_conf = (flow_conf - flow_conf.min()) / (flow_conf.max() - flow_conf.min() + 1e-6)

        # Optional masking
        if mask is not None:
            depth_conf = depth_conf * mask
            flow_conf = flow_conf * mask

        total_conf = depth_conf + flow_conf + 1e-6
        fused = (depth_map * depth_conf + flow_mag * flow_conf) / total_conf
        frame_conf = float(np.mean(total_conf))

        return fused, frame_conf
