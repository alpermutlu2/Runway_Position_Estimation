
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

def launch_streamlit_gui(outputs):
    st.set_page_config(layout='wide')
    st.title('🔍 Visual Localization Dashboard')

    num_frames = len(outputs['depth'])
    is_live = st.checkbox('Simulate Live Feed')
    log_messages = []

    if is_live:
        play_button = st.button('▶ Start Live Stream')
        frame_idx = 0
        if play_button:
            for frame_idx in range(num_frames):
                st.experimental_rerun()

    selected_frame = st.slider('Select Frame', 0, num_frames - 1, 0)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Depth Map')
        depth = outputs['depth'][selected_frame]
        fig1, ax1 = plt.subplots()
        im = ax1.imshow(depth, cmap='plasma')
        fig1.colorbar(im, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader('2D Pose Trajectory')
        poses = outputs['pose']
        xs = [pose[0, 3] for pose in poses]
        ys = [pose[1, 3] for pose in poses]
        fig2, ax2 = plt.subplots()
        ax2.plot(xs, ys, '-o', label='Trajectory')
        ax2.plot(xs[selected_frame], ys[selected_frame], 'ro', label='Selected Frame')
        ax2.set_aspect('equal')
        ax2.legend()
        st.pyplot(fig2)

    if 'evaluation' in outputs:
        st.markdown('### 📊 Evaluation Metrics')
        for key, value in outputs['evaluation'].items():
            st.metric(label=key, value=f"{value:.4f}")

    st.markdown('---')
    st.text(f"Pose Matrix for Frame {selected_frame}:")
    st.code(str(outputs['pose'][selected_frame]))

    st.markdown('---')
    st.markdown('### 🧾 Log Panel')
    log_messages.append(f"Viewing frame {selected_frame} at {time.strftime('%H:%M:%S')}")
    for log in log_messages[-5:][::-1]:
        st.text(log)
