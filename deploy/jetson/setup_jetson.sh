
#!/bin/bash
echo "Setting up Jetson deployment environment..."
sudo apt update
sudo apt install -y python3-pip python3-opencv libopenblas-dev
pip3 install torch torchvision onnxruntime numpy opencv-python

echo "Exporting MiDaS model to ONNX..."
python3 deploy/export_midas_onnx.py

echo "Deployment ready. Run: python3 jetson_inference_runner.py"
