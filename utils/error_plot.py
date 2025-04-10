
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_ate_trend(errors):
    fig, ax = plt.subplots()
    ax.plot(errors, color='red', label='ATE (m)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('ATE')
    ax.set_title('Absolute Trajectory Error Over Time')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def plot_error_heatmap(gt_poses, est_poses):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    gt = np.array(gt_poses)
    est = np.array(est_poses)
    if len(gt) != len(est): return

    errors = np.linalg.norm(gt[:, :3, 3] - est[:, :3, 3], axis=1)
    norm = Normalize(vmin=errors.min(), vmax=errors.max())
    cmap = cm.get_cmap('hot')

    fig, ax = plt.subplots()
    for i in range(len(gt)):
        ax.scatter(gt[i, 0, 3], gt[i, 2, 3], color=cmap(norm(errors[i])))
    ax.set_title("Trajectory Error Heatmap")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.axis("equal")
    st.pyplot(fig)
