import torch
from models.depth_net import ProbabilisticDepthNet

model = ProbabilisticDepthNet().eval()
dummy_input = torch.randn(1, 3, 256, 512, device="cuda")
torch.onnx.export(
    model,
    dummy_input,
    "depth_net.onnx",
    input_names=["input"],
    output_names=["depth_mean", "depth_var"],
    dynamic_axes={"input": {0: "batch"}, "depth_mean": {0: "batch"}, "depth_var": {0: "batch"}},
)