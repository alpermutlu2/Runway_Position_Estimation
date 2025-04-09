# Getting Started

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/alpermutlu2/aysezeynepahmet.git

   pip install -r requirements.txt

   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h.pth

   python train.py --config configs/kitti.yaml

   
---

### **Integration Workflow**  
1. **Training**:  
   - Uses SAM to mask dynamic objects during training.  
   - Uncertainty loss improves robustness in ambiguous regions.  

2. **Inference**:  
   - Export to TensorRT for real-time edge deployment.  
   - ROS node publishes depth maps for robotic applications.  

3. **Dynamic Handling**:  
   - SAM masks are applied during both training and inference to ignore moving objects.  

---

### **Next Steps**  
1. **Hybrid NeRF-SLAM**: Extend `models/nerf.py` to fuse SLAM poses with NeRF rendering.  
2. **Quantization**: Add post-training quantization in `inference/tensorrt/quantize.py`.  
3. **Benchmarking**: Add scripts to compare against KITTI leaderboard methods.  

Let me know if you’d like to expand on any subsystem!