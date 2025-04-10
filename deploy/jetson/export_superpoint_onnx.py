
import torch
import torch.nn as nn

class DummySuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 65, 1)
        )

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    model = DummySuperPoint()
    dummy_input = torch.randn(1, 1, 240, 320)
    torch.onnx.export(model, dummy_input, "superpoint.onnx",
                      input_names=["input"], output_names=["output"],
                      opset_version=11)
    print("✅ Exported dummy SuperPoint to superpoint.onnx")
