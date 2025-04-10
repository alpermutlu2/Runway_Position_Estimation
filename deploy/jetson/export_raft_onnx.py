
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.flow_head = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, image_pair):
        feat = self.encoder(image_pair)
        flow = self.flow_head(feat)
        return flow

if __name__ == "__main__":
    model = DummyRAFT()
    dummy_input = torch.randn(1, 6, 224, 320)  # Concatenated image pair (2x3 channels)
    torch.onnx.export(model, dummy_input, "raft.onnx",
                      input_names=["image_pair"], output_names=["flow"],
                      opset_version=11)
    print("✅ Exported simplified RAFT to raft.onnx")
