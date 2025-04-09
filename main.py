
import os
import torch
import numpy as np
from core.inference import run_inference_pipeline
from core.bundle_adjustment import run_bundle_adjustment
from utils.temporal_filter import TemporalFilter
from utils.confidence_fusion import fuse_depth_maps_with_confidence
from utils.flow_depth_mask import compute_flow_depth_mask
from evaluation.metrics import evaluate_trajectory
from export.kitti_exporter import export_trajectory_kitti_format
from visualization.viewer import launch_streamlit_gui
from slam.slam_runner import run_slam


def load_data_from_npy(image_path, depth_path, pose_path):
    image = np.load(image_path)
    depth = np.load(depth_path)
    pose = np.loadtxt(pose_path)
    return image, depth, pose


def main(config):
    # Load data and run model inference
    if config.get('use_dataset', False):
        image_path = os.path.join(config['input_path'], 'images/train/img_000.npy')
        depth_path = os.path.join(config['input_path'], 'depth_gt/train/depth_000.npy')
        pose_path = os.path.join(config['input_path'], 'poses_gt/train/pose_000.txt')
        image, depth, pose = load_data_from_npy(image_path, depth_path, pose_path)
        outputs = {
            'depth_sources': [depth],
            'confidence_maps': [np.ones_like(depth)],
            'depth': depth,
            'pose': [pose] * 10,
            'flow': [np.random.rand(*depth.shape, 2)] * 9,
            'landmarks': [np.random.rand(3) for _ in range(10)],
            'observations': {(i, i): np.array([32, 32]) for i in range(10)}
        }
    else:
        outputs = run_inference_pipeline(config['input_path'], config['models'])

    # Apply flow-depth inconsistency mask
    if config.get('enable_flow_depth_mask', False):
        masked_depths = []
        for i in range(len(outputs['depth']) - 1):
            mask = compute_flow_depth_mask(
                outputs['flow'][i], outputs['depth'][i], outputs['depth'][i + 1]
            )
            masked_depth = outputs['depth'][i] * mask
            masked_depths.append(masked_depth)
        outputs['depth'] = masked_depths

    # Apply temporal filtering
    if config.get('enable_temporal_filter', False):
        tf = TemporalFilter(method='ema', alpha=config.get('filter_alpha', 0.9))
        outputs['depth'] = tf.apply(outputs['depth'])
        outputs['pose'] = tf.apply(outputs['pose'])

    # Confidence-weighted fusion of depth sources
    if config.get('enable_confidence_fusion', False):
        outputs['depth'] = fuse_depth_maps_with_confidence(
            outputs['depth_sources'], outputs['confidence_maps']
        )

    # Optional Bundle Adjustment
    if config.get('use_bundle_adjustment', False):
        outputs['pose'] = run_bundle_adjustment(
            outputs['pose'], outputs['landmarks'], outputs['observations']
        )

    # Evaluate using ATE and RPE
    results = evaluate_trajectory(outputs['pose'], config['ground_truth'])
    print("Evaluation Results:", results)

    # Export pose trajectory
    export_trajectory_kitti_format(outputs['pose'], config['export_path'])

    # Optional Streamlit GUI
    if config.get('enable_gui', False):
        launch_streamlit_gui(outputs)

    if config.get('run_slam', False):

    if config.get('export_pointcloud', False):
        from utils.pointcloud_export import export_pointcloud
        for i, depth in enumerate(outputs['depth']):
            export_pointcloud(depth, outputs['pose'][i], save_path=f"export/cloud_{i:03d}.ply")
        run_slam(outputs['pose'])


if __name__ == '__main__':
    config = {
        'input_path': 'data',
        'models': ['M4Depth', 'CoDEPS'],
        'ground_truth': 'data/groundtruth_01.txt',
        'enable_flow_depth_mask': True,
        'enable_temporal_filter': True,
        'filter_alpha': 0.9,
        'enable_confidence_fusion': True,
        'use_bundle_adjustment': True,
        'enable_gui': False,
        'use_dataset': True,
        'run_slam': True,
        'export_path': 'export/kitti/sequence_01/'
    }
    main(config)
