
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(gt=None, est=None):
    fig, ax = plt.subplots()
    if gt is not None:
        gt = np.array(gt)
        ax.plot(gt[:, 0], gt[:, 2], label='Ground Truth', color='blue')
    if est is not None:
        est = np.array(est)
        ax.plot(est[:, 0], est[:, 2], label='Estimated', color='red')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
