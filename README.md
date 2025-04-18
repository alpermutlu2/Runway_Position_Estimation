# 🛬  Runway Position Estimation System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![OpenCV Version](https://img.shields.io/badge/opencv-4.5%2B-orange)](https://opencv.org/)

---

## 📌 Key Features

![Runway Position Estimation Pipeline](./assets/runway_pipeline_diagram.png)


- **Semantic Runway Detection** — via Segment Anything (SAM) with early rejection.
- **Vanishing Point Estimation** — using RANSAC-based line fitting to estimate heading angle (yaw).
- **Hybrid Tracking** — ORB-SLAM3 fallback with LightGlue + SuperPoint frontend.
- **Depth & Flow Fusion** — fuses M4Depth / MiDaS and RAFT optical flow for 3D position.
- **Temporal Filtering** — smooths estimates with a custom pose filter.
- **Loop Closure** — optional relocalization and pose graph optimization.
- **Supports** image sequences, video files, webcam (real-time), CPU or GPU.
- **Deployment** on Hugging Face Spaces and Google Colab.

---

## Installation

### Prerequisites
- Python 3.7+
- OpenCV 4.5+
- NumPy
- Matplotlib (for visualization)

### Quick Install
```bash
pip install -r requirements.txt

git clone https://github.com/alpermutlu2/Runway_Position_Estimation.git
cd Runway_Position_Estimation
pip install -e .
```


### 🚀 Quick Start


```bash
git clone https://github.com/alpermutlu2/Runway_Position_Estimation.git
cd Runway_Position_Estimation
bash setup.sh
```

---

#### Usage Examples
```markdown
## Usage

### Basic Command Line
```bash
python runway_estimator.py --input path/to/image.jpg --output results/

from runway_estimator import RunwayEstimator

estimator = RunwayEstimator(
    camera_params="config/calibration.json",
    edge_thresholds=(50, 150)
)
result = estimator.process_image("input.jpg")
```

---

#### Configuration Template
Make `config/calibration.json`:
```json
{
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_coeffs": [k1, k2, p1, p2, k3],
    "runway_width_meters": 45.0
}
```
---

## Methodology

### Processing Pipeline
1. **Preprocessing**
   - Adaptive histogram equalization
   - Noise reduction (bilateral filtering)
   
2. **Feature Detection**
   - Canny edge detection
   - Probabilistic Hough lines
   
---

### 🧪 2. Run on a Video

```bash
python main.py \
    --video assets/sample_runway.mp4 \
    --realtime \
    --visualize \
    --save
```

You can also use `--image_dir` for KITTI-style image folders or leave both empty to run on dummy frames.

---

## 🌐 Try Online

- **🤗 Hugging Face Space**: [Runway Estimation App](https://huggingface.co/spaces/your_username/runway-estimation)  
- **🧪 Colab Demo**: [Open Notebook](https://colab.research.google.com/github/alpermutlu2/Runway_Position_Estimation/blob/main/Runway_Position_Estimation_Demo.ipynb)

---

## ⚙️ Arguments

| Flag | Description |
|------|-------------|
| `--video` | Path to input video |
| `--image_dir` | Path to image sequence |
| `--realtime` | Enable real-time visualization |
| `--save` | Save output video |
| `--device` | `cuda` or `cpu` |
| `--visualize` | Draw debug overlays |
| `--gt_path` | KITTI-style GT pose file |
| `--enable_loop_closure` | Enables relocalization and graph optimization |
| `--skip_weight_check` | Skip download check on boot |

---

## 📁 Directory Structure

```
Runway_Position_Estimation/
├── main.py
├── setup.sh
├── requirements.txt
├── Runway_Position_Estimation_Demo.ipynb
├── tracking/
├── depth/
├── inference/
├── evaluation/
├── optimization/
├── utils/
└── scripts/
```

---

## 📊 Evaluation

Supports KITTI-format trajectory logging, ATE computation, and keyframe optimization. Compatible with TUM/KITTI visual SLAM benchmarks.

---

## 🧠 Credits

Developed by [@alpermutlu2](https://github.com/alpermutlu2)  
Includes components inspired by MiDaS, LightGlue, ORB-SLAM3, RAFT, SAM, and more.

---

## 📝 License

MIT License (c) 2024-present. See `LICENSE`.