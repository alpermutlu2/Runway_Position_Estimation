
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGlueONNX(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)
        self.attn_proj = nn.Linear(input_dim, input_dim)

    def forward(self, desc0, desc1):
        q = self.linear_q(desc0)  # [B, N, D]
        k = self.linear_k(desc1)  # [B, M, D]
        v = self.linear_v(desc1)

        attn = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        weights = F.softmax(attn, dim=-1)
        out = torch.bmm(weights, v)
        return self.attn_proj(out)

if __name__ == "__main__":
    model = LightGlueONNX()
    dummy_desc0 = torch.randn(1, 128, 256)  # [batch, N, D]
    dummy_desc1 = torch.randn(1, 256, 256)  # [batch, M, D]
    torch.onnx.export(model, (dummy_desc0, dummy_desc1), "lightglue.onnx",
                      input_names=["desc0", "desc1"], output_names=["matched"],
                      opset_version=11)
    print("✅ Exported LightGlue (ONNX-compatible) to lightglue.onnx")
