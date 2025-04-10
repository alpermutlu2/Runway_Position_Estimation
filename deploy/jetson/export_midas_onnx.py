
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
dummy = torch.randn(1, 3, 384, 384)
torch.onnx.export(model, dummy, "midas.onnx", input_names=['input'], output_names=['output'])
print("Exported MiDaS to midas.onnx")
