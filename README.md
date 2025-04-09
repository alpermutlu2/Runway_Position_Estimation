
# Runway Position Estimation

This is an enhanced version of the original monocular visual localization repository with full integration of modular components.

## ✅ Features

- Temporal filtering (EMA)
- Confidence-based depth fusion
- Flow-depth inconsistency masking
- Bundle adjustment
- Evaluation toolkit (ATE, RPE)
- KITTI trajectory export
- Streamlit GUI
- Modular model handling via ModelWrapper

## 🗂 Folder Structure

- `core/` — core algorithms (inference, BA)
- `utils/` — helper modules (filtering, masking, fusion)
- `models/` — wrappers for external models like M4Depth, CoDEPS
- `evaluation/` — trajectory evaluation scripts
- `export/` — exporters for formats like KITTI
- `visualization/` — GUI tools
- `scripts/` — runnable scripts
- `data/` — expected input data folders

## 🚀 Running

```bash
python main.py
```

## 🧪 Testing

```bash
python scripts/test_pipeline.py
```

## 📦 Configuration

See `main.py` for editable config options.

MIT License


## 🧪 Data Preparation

This repository supports structured training/validation/testing datasets:

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── depth_gt/
│   ├── train/
│   ├── val/
│   └── test/
└── poses_gt/
    ├── train/
    ├── val/
    └── test/
```

Each sample includes:
- `images/*.npy` : RGB input
- `depth_gt/*.npy` : Ground-truth depth maps
- `poses_gt/*.txt` : 3x4 pose matrices

### 🔀 Dataset Split Script

To automatically split your training set:

```bash
python scripts/split_dataset.py
```

This will shuffle and copy files into `train`, `val`, and `test` folders based on the default split ratio (70/15/15).


## 🧭 Visual SLAM Modules

This project includes foundational components for Visual SLAM integration:

```
slam/
├── keyframe_manager.py        # Decides when to add new keyframes
├── pose_graph.py              # Manages a graph of poses and edges
├── local_bundle_adjustment.py # Performs local graph refinement
└── slam_runner.py             # End-to-end SLAM pipeline using keyframes
```

### 🔁 Running SLAM Module

To test SLAM pipeline with dummy pose data:

```bash
python slam/slam_runner.py
```

### 📌 Key Concepts

- **Keyframe Selection**: Selects poses based on movement threshold.
- **Pose Graph**: Stores relative transformations between poses.
- **Bundle Adjustment**: Placeholder for local optimization over pose graph.

In future versions, global loop closure and real-time tracking can be added.


## 🔁 Updated Pipeline Configuration

`main.py` now supports modular integration through the following config flags:

| Config Key               | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| `use_dataset`            | Load data from `.npy` and `.txt` files under `data/`         |
| `run_slam`               | Run the Visual SLAM module using output poses                |
| `enable_gui`             | Enable Streamlit GUI                                         |
| `enable_temporal_filter`| Apply EMA smoothing to pose and depth                        |
| `enable_confidence_fusion`| Fuse multiple depth maps with confidence weights           |
| `use_bundle_adjustment` | Refine pose via landmark reprojection                        |
| `enable_flow_depth_mask`| Mask dynamic regions using depth/flow consistency            |

---

## 🧭 Visual SLAM Integration

After running `main.py` with `run_slam=True`, the following occurs:
- Keyframe selection based on pose displacement
- Pose graph construction using relative transforms
- Optional local bundle adjustment for keyframe refinement

> ✅ Upcoming improvements: pose graph optimization with external solver

---

## 🌍 Optional Point Cloud Integration (Future)

This system can optionally be extended to produce:
- Depth-based 3D point clouds per frame
- Global point cloud fusion using SLAM poses
- Export to `.ply` or `.pcd` formats

---

## 🧪 Model Training/Fine-Tuning (Planned)

To allow fine-tuning with your own data:
- Extend the `models/` wrapper to include training logic
- Create `train.py` for supervised or self-supervised training
- Enable augmentation and checkpoint saving

Example starter training configs will be added in future iterations.


---

## 🔁 SLAM Optimization Integration

The pose graph optimization routine is now embedded in `slam/pose_graph.py`:
- Adds mock optimization (random perturbation for demo)
- Future integration with Ceres/g2o possible
- Triggered automatically when `run_slam=True`

---

## 🌍 Point Cloud Export (Optional)

You can generate `.ply` point cloud files per-frame using:

```python
'export_pointcloud': True
```

This uses `utils/pointcloud_export.py` and saves files to:

```
export/cloud_000.ply
export/cloud_001.ply
...
```

---

## 🧪 Training & Fine-Tuning Pipeline

Train a simple depth prediction model using `.npy` input with:

```bash
python train.py
```

Requirements:
- Images in `data/images/train/*.npy` (shape: HxWx3, dtype uint8 or float32)
- Depths in `data/depth_gt/train/*.npy` (shape: HxW, dtype float32)

Trains a small convolutional model using L1 loss, saves to:

```
checkpoints/depth_model.pth
```

---

## 🧠 Overall Integrated Pipeline

All modules are now interconnected via `main.py`, which supports:

| Module                    | Config Key             | Notes                          |
|---------------------------|------------------------|---------------------------------|
| Inference from models     | `models`               | M4Depth, CoDEPS wrappers       |
| Dataset input             | `use_dataset`          | From `.npy`/`.txt` under `data/`|
| Flow-depth masking        | `enable_flow_depth_mask`| Optional, dynamic rejection    |
| Temporal smoothing        | `enable_temporal_filter`| Uses EMA                       |
| Confidence fusion         | `enable_confidence_fusion`| Fuse multiple depths         |
| Bundle adjustment         | `use_bundle_adjustment`| Refines pose from observations |
| SLAM tracking             | `run_slam`             | Adds pose graph + keyframes    |
| Pose graph optimization   | Auto in SLAM           | Basic graph optimization       |
| Point cloud export        | `export_pointcloud`    | One file per frame             |
| GUI visualization         | `enable_gui`           | Streamlit interface            |



---

## 🖼 Advanced Streamlit GUI Features

Launch the dashboard using:

```bash
streamlit run main.py
```

Features:
- 🔘 Frame slider for previewing any frame
- 🎞 Simulated live playback via checkbox
- 🌄 Depth map visualized with colorbar
- 🧭 2D trajectory plot with pose highlights
- 📊 Evaluation metrics shown per run
- 💬 Live log panel shows recent frame views

> Optionally extend to Open3D 3D viewer in future versions
