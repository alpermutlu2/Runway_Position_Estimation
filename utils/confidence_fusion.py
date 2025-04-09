
import numpy as np

def fuse_depth_maps_with_confidence(depth_maps, confidence_maps):
    fused = np.zeros_like(depth_maps[0])
    total_confidence = np.zeros_like(confidence_maps[0])
    for depth, conf in zip(depth_maps, confidence_maps):
        fused += depth * conf
        total_confidence += conf
    total_confidence[total_confidence == 0] = 1e-8
    return fused / total_confidence
