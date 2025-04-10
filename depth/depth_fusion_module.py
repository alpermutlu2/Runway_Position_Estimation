
import numpy as np
from depth.midas_depth import MiDaSDepth
from depth.m4depth_wrapper import M4Depth
from flow.raft_wrapper import RAFTFlow
from fusion.depth_flow_fusion import DepthFlowFusion
from segmentation.semantic_filter import SemanticFilter

class DepthFusionModule:
    def __init__(self, config=None):
        print("Full fusion depth module initialized.")
        self.midas = MiDaSDepth()
        self.m4depth = M4Depth()
        self.raft = RAFTFlow()
        self.fuser = DepthFlowFusion()
        self.masker = SemanticFilter()
        self.last_image = None

    def fuse(self, image, keypoints, pose):
        # Get semantic mask
        mask = self.masker.get_mask(image)

        # Get MiDaS and M4Depth maps
        midas_depth = self.midas.predict(image)
        m4depth = self.m4depth.predict(image)

        # Use RAFT only if we have previous frame
        if self.last_image is not None:
            flow_map = self.raft.compute_flow(self.last_image, image)
        else:
            flow_map = np.zeros((image.shape[0], image.shape[1], 2))

        # Store current frame
        self.last_image = image.copy()

        # Naively average MiDaS + M4Depth
        depth_avg = (midas_depth + m4depth) / 2.0

        # Fuse with flow
        fused_depth = self.fuser.fuse(depth_avg, flow_map, mask)

        # Estimate position as pose translation with noise (placeholder)
        position_3d = pose[:3, 3] + np.random.normal(0, 0.01, 3)

        return fused_depth, position_3d
