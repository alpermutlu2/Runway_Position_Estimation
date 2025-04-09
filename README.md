# aysezeynepahmet

Monocular depth estimation and visual localization system featuring semantic segmentation, uncertainty modeling, and evaluation on standard benchmarks. Designed with modularity in mind to enable experimentation, fusion, and deployment in real-time or academic settings.

## 🚀 Project Overview

This project implements a monocular depth estimation and scale recovery framework, built on top of cutting-edge deep learning techniques. It includes segmentation (SAM), depth fusion, multi-view optimization, uncertainty modeling, and robust evaluation.

### ✅ Features
- **Monocular Depth Estimation** using custom CNN or Transformer backbones
- **Dynamic Object Masking** with SAM (Segment Anything Model) to ignore moving objects
- **Depth Fusion**: Combines outputs from multiple models or views using confidence-aware techniques
- **Uncertainty Modeling**: Probabilistic outputs enable confidence-based reasoning
- **SLAM-Compatible**: Designed to integrate into SLAM pipelines
- **Dataset Agnostic**: Works with KITTI, NYU, and custom formats
- **Evaluation Suite** with metrics: AbsRel, RMSE, δ1, δ2, δ3
- **Visualization Tools** for depth maps, uncertainty, flow, and object masks
- **Modular Architecture**: Easy to plug in new models, loss functions, or datasets

---

## 📁 Project Structure

```bash
aysezeynepahmet/
├── core/              # Backbone logic, registration, utilities
├── data/              # Dataset loaders, augmentations
├── depth/             # Depth estimation architectures
├── docs/              # Additional documentation and references
├── evaluation/        # Metric computation and evaluation tools
├── frontends/         # Interfaces for model/data/SLAM input
├── fusion/            # Depth fusion and probabilistic reasoning
├── inference/         # Forward pass and prediction pipelines
├── losses/            # Custom loss functions
├── models/            # Model registration and architecture configs
├── outputs/logs/      # Training logs, checkpoints, and results
├── scripts/           # Utility scripts (demo, preprocess, etc.)
├── segmentation/      # SAM integration and semantic/instance segmentation
├── slam/              # SLAM-related integration modules
├── tracking/          # Object tracking helpers (optical flow, etc.)
├── utils/             # General helpers, logging, time, I/O
├── visualization/     # Depth visualizers, flow viewers, etc.
├── configs/           # YAML config files for models and training
├── main.py            # Entry point for complete pipeline
├── train.py           # Core training script
├── train_cli.py       # CLI wrapper for flexible training
├── visualize_depth.py # Visualize predicted depth maps
├── requirements.txt   # Python dependencies
├── setup.py           # Installation script
├── LICENSE            # MIT License
├── README.md          # This file
├── Benchmark_Report.pdf # Evaluation results
```

---

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/alpermutlu2/aysezeynepahmet.git
cd aysezeynepahmet
```

2. **Create and activate a virtual environment** *(optional but recommended)*:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install the package**:
```bash
python setup.py install
```

---

## ⚙️ Usage

### 🔨 Train a Model
```bash
python train_cli.py --config configs/train_config.yaml
```

### 📈 Evaluate a Model
```bash
python evaluation/evaluate.py \
    --model_path checkpoints/my_model.pth \
    --data_path data/kitti_raw
```

### 👁️ Visualize Depth Prediction
```bash
python visualize_depth.py \
    --model_path checkpoints/my_model.pth \
    --image_path test_images/sample.png
```

### 🧪 Run Full Inference Pipeline
```bash
python main.py --config configs/full_pipeline.yaml
```

---

## 🧾 Configuration

All models, datasets, and training logic are configured through YAML files in the `configs/` folder.

Example keys:
```yaml
model:
  type: resnet18
  pretrained: true
  channels: 64

dataset:
  name: kitti
  path: ./data/kitti
  img_size: [320, 1024]
  augment: true

training:
  epochs: 30
  batch_size: 4
  lr: 0.0001
  loss: scale_invariant
```

---

## 📊 Metrics

The evaluation suite computes:
- **AbsRel**: Absolute Relative Difference
- **RMSE**: Root Mean Square Error
- **δ1 / δ2 / δ3**: Accuracy under thresholds 1.25, 1.25², 1.25³

These are implemented in `evaluation/metrics.py` and can be extended for your own use cases.

---

## 🧠 Models

Supported backbones include:
- ResNet (18, 34, 50)
- ViT-style depth predictors
- Lightweight MobileNet versions

You can define custom models by adding them to `models/` and registering via `core/registry.py`.

---

## 🖼️ Visualization Tools

Visual diagnostics are available for:
- Depth maps
- Error maps
- Semantic masks
- Confidence maps (uncertainty)

These help understand failure cases and debug model behavior.

---

## 📂 Supported Datasets

- **KITTI Raw** and **KITTI Eigen split**
- **NYU Depth V2**
- Custom formats via `data/custom_loader.py`

---

## 🤖 Inference + SLAM Integration

Outputs can be structured to match formats required by SLAM pipelines (e.g., TUM, KITTI). Add your frontend in `frontends/` and plug into `main.py`.

---

## 📌 Advanced Topics

- **SAM Integration**: Segment Anything Model is used to generate masks for moving objects.
- **Fusion Techniques**: Combine M4Depth + MiDaS + RAFT with confidence weighting.
- **Uncertainty Estimation**: Bayesian modeling or dropout-based approximations supported.
- **Tracking**: Optional optical flow support (RAFT) for tracking scene dynamics.

---

## 🧩 Extending the Codebase

To add a new model:
1. Define architecture in `models/new_model.py`
2. Register it in `core/registry.py`
3. Add to YAML config

To add a new dataset:
1. Create loader in `data/new_dataset.py`
2. Add transform pipeline
3. Register loader in `core/loader_factory.py`

---

## 🤝 Contributing

We welcome contributions! Feel free to:
- Submit issues and bug reports
- Improve documentation
- Add new models or fusion techniques

See `CONTRIBUTING.md` for full guidelines.

---

## 📄 License

Licensed under the MIT License. See the `LICENSE` file for details.

---

For benchmarks, sample outputs, and diagrams, see `Benchmark_Report.pdf`.

Happy depth mapping! 🌍📸
