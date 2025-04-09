
from main import main

if __name__ == '__main__':
    config = {
        'input_path': 'data/sequence_01',
        'models': ['M4Depth', 'CoDEPS'],
        'ground_truth': 'data/groundtruth_01.txt',
        'enable_flow_depth_mask': True,
        'enable_temporal_filter': True,
        'filter_alpha': 0.9,
        'enable_confidence_fusion': True,
        'use_bundle_adjustment': True,
        'enable_gui': False,
        'export_path': 'export/kitti/sequence_01/'
    }
    main(config)
