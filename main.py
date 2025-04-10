# main.py — Full Real-time & Batch Visual Localization Pipeline

import argparse
import os
import numpy as np
import cv2
import torch

from tracking.hybrid_tracker import HybridTracker
from depth.depth_fusion_module import DepthFusionModule
from filtering.pose_filter import PoseFilter
from evaluation.evaluator import Evaluator
from evaluation.kitti_logger import KITTILogger
from evaluation.kitti_metrics import load_kitti_poses, compute_ATE
from data.image_sequence_loader import ImageSequenceLoader
from utils.video_io import VideoStreamer
from optimization.keyframe_logger import KeyframeLogger
from optimization.relocalization_detector import RelocalizationDetector
from optimization.pose_graph_optimizer import PoseGraphOptimizer
from scripts.check_weights import check_weights
from inference.segmentation import RunwaySegmenter
from inference.preprocess import apply_clahe
from utils.profiler import Timer
from inference.line_detection import detect_runway_lines_ransac
from utils.visualization import overlay_trajectory, draw_frame_info


def estimate_yaw_from_lines(left_line, right_line, K):
    if left_line is None or right_line is None:
        return None
    m1, b1 = left_line
    m2, b2 = right_line
    if abs(m1 - m2) < 1e-3:
        return None
    x_v = (b2 - b1) / (m1 - m2)
    y_v = m1 * x_v + b1
    vp_homog = np.array([x_v, y_v, 1.0])
    v_cam = np.linalg.inv(K).dot(vp_homog)
    yaw = np.arctan2(v_cam[0], v_cam[2])
    return np.degrees(yaw)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str, default='hybrid')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--gt_path', type=str, default='')
    parser.add_argument('--enable_loop_closure', action='store_true')
    parser.add_argument('--skip_weight_check', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_weight_check:
        if not check_weights():
            print("❌ Weight check failed. Exiting.")
            return

    config = {'orbslam3_min_quality': 0.3}
    tracker = HybridTracker(config)
    depth_module = DepthFusionModule(config)
    pose_filter = PoseFilter(alpha=0.8)
    evaluator = Evaluator()
    logger = KITTILogger()
    keyframe_log = KeyframeLogger()
    loop_detector = RelocalizationDetector()
    graph_optimizer = PoseGraphOptimizer()
    segmenter = RunwaySegmenter(device=args.device)

    timer = Timer()
    est_poses = []
    loop_closed = False

    if args.video:
        loader = VideoStreamer(args.video)
        use_video = True
    elif args.image_dir and os.path.exists(args.image_dir):
        loader = ImageSequenceLoader(args.image_dir)
        use_video = False
    else:
        print("⚠️ No input provided. Using simulated dummy frames.")
        loader = None
        use_video = False

    if args.save and loader:
        output_path = 'output.avi'
        out_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            loader.fps() if use_video else 30,
            loader.resolution() if use_video else (640, 480)
        )
    else:
        out_writer = None

    for frame_id in range(10 if loader is None else len(loader)):
        if loader:
            image, timestamp = loader.get_next()
            if image is None:
                break
        else:
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            timestamp = frame_id / 30.0

        timer.start("CLAHE")
        image = apply_clahe(image)
        timer.stop("CLAHE")

        timer.start("Segmentation")
        mask = segmenter.predict(image)
        timer.stop("Segmentation")

        runway_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        if runway_ratio < 0.01:
            print(f"[Frame {frame_id}] Rejected: no visible runway.")
            continue

        timer.start("RANSAC Lines")
        left_line, right_line, edge_viz = detect_runway_lines_ransac(image, mask)
        timer.stop("RANSAC Lines")

        K = np.array([[600, 0, image.shape[1] / 2], [0, 600, image.shape[0] / 2], [0, 0, 1]])
        yaw = estimate_yaw_from_lines(left_line, right_line, K)
        if yaw is not None:
            print(f"[Frame {frame_id}] Estimated Heading Yaw: {yaw:.2f}°")

        timer.start("Tracking")
        pose, keypoints, is_keyframe = tracker.track(image, timestamp)
        timer.stop("Tracking")

        timer.start("Depth Fusion")
        depth_map, position_3d = depth_module.fuse(image, keypoints, pose)
        timer.stop("Depth Fusion")

        timer.start("Filtering")
        filtered_pos = pose_filter.filter(position_3d)
        timer.stop("Filtering")

        evaluator.log(frame_id, pose, filtered_pos)
        logger.log(pose)
        est_poses.append(pose)

        if is_keyframe:
            keyframe_log.log(frame_id, pose)
            if args.enable_loop_closure:
                loop = loop_detector.check_loop(pose, keyframe_log.get_recent(10))
                if loop and not loop_closed:
                    print(f"[Loop Closure Detected] at frame {frame_id}")
                    optimized = graph_optimizer.optimize(keyframe_log.get_all_poses())
                    est_poses = optimized
                    loop_closed = True

        if args.visualize or args.realtime:
            debug_frame = image.copy()
            h = debug_frame.shape[0]

            def draw_line(line, color):
                if line is not None:
                    slope, intercept = line
                    x1 = int((h - intercept) / slope)
                    x2 = int((0 - intercept) / slope)
                    cv2.line(debug_frame, (x1, h), (x2, 0), color, 2)

            draw_line(left_line, (0, 255, 0))
            draw_line(right_line, (0, 0, 255))
            overlay_trajectory(debug_frame, [p[:3, 3] for p in est_poses])
            draw_frame_info(debug_frame, frame_id, args.device)
            cv2.imshow("Runway Localization", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.save and out_writer:
            out_writer.write(debug_frame)

        print(f"[Frame {frame_id}] Filtered Position: {filtered_pos}, Keyframe: {is_keyframe}")

    tracker.shutdown()
    timer.report()
    evaluator.summarize()
    logger.save()

    if args.gt_path and os.path.exists(args.gt_path):
        gt_poses = load_kitti_poses(args.gt_path)
        ate = compute_ATE(gt_poses, est_poses)
        print(f"ATE against GT: {ate:.4f} meters")

    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
