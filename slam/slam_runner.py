
from slam.keyframe_manager import KeyframeManager
from slam.pose_graph import PoseGraph
from slam.local_bundle_adjustment import local_bundle_adjustment

def run_slam(pose_sequence):
    kf_manager = KeyframeManager(threshold=0.5)
    graph = PoseGraph()

    last_kf_pose = None
    for idx, pose in enumerate(pose_sequence):
        if last_kf_pose is None or kf_manager.should_add_keyframe(pose, last_kf_pose):
            kf_manager.add_keyframe(pose, idx)
            graph.add_pose(idx, pose)
            last_kf_pose = pose

            if idx > 0:
                graph.add_edge(idx-1, idx, pose @ pose_sequence[idx-1])

    local_bundle_adjustment([pose for _, pose in kf_manager.keyframes], {})
    print(f"SLAM complete with {len(kf_manager.keyframes)} keyframes.")

if __name__ == '__main__':
    import numpy as np
    dummy_poses = [np.eye(4) for _ in range(10)]
    for i in range(10):
        dummy_poses[i][:3, 3] = [i * 0.6, 0, 0]
    run_slam(dummy_poses)
