
import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.image_sequence_loader import ImageSequenceLoader
from tracking.hybrid_tracker import HybridTracker
from depth.depth_fusion_module import DepthFusionModule
from filtering.pose_filter import PoseFilter

st.set_page_config(page_title="Runway Position Estimation", layout="wide")

st.title("📍 Runway Position Estimation Visualizer")

image_dir = st.text_input("Path to Image Sequence", value="sample_data/")
run_button = st.button("Run Pipeline")

if run_button and os.path.exists(image_dir):
    loader = ImageSequenceLoader(image_dir)
    tracker = HybridTracker({'orbslam3_min_quality': 0.3})
    depth_module = DepthFusionModule()
    pose_filter = PoseFilter()

    st_frame = st.empty()
    st_depth = st.empty()
    st_position = st.empty()
    st_traj = st.empty()

    trajectory = []

    for frame_id in range(len(loader)):
        image, timestamp = loader.get_next()
        if image is None:
            break

        pose, keypoints, is_keyframe = tracker.track(image, timestamp)
        depth_map, position_3d = depth_module.fuse(image, keypoints, pose)
        filtered_pos = pose_filter.filter(position_3d)
        trajectory.append(filtered_pos)

        # Display frame
        st_frame.image(image, caption=f"Frame {frame_id}", channels="BGR", use_column_width=True)

        # Display depth
        fig, ax = plt.subplots()
        ax.imshow(depth_map, cmap='viridis')
        ax.set_title("Estimated Depth")
        ax.axis("off")
        st_depth.pyplot(fig)

        # Display position
        st_position.text(f"Estimated Position (filtered): {filtered_pos.round(3)}")

        # Plot trajectory
        traj_np = np.array(trajectory)
        fig2, ax2 = plt.subplots()
        ax2.plot(traj_np[:, 0], traj_np[:, 2], marker='o')  # X-Z plane
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Z [m]")
        ax2.set_title("Trajectory (X-Z)")
        ax2.grid(True)
        st_traj.pyplot(fig2)

    tracker.shutdown()
else:
    st.warning("Please enter a valid image sequence path and press 'Run Pipeline'.")
