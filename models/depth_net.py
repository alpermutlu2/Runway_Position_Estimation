
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ProbabilisticDepthNet(nn.Module):
    def __init__(self, backbone="resnet18", min_depth=0.1, max_depth=100.0):
        super().__init__()

        # Depth range
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Backbone initialization
        if backbone == "resnet18":
            encoder = resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(encoder.children())[:-2])
            self.encoder_out_channels = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

        # Depth head (mean and log variance)
        self.depth_head = nn.Conv2d(self.encoder_out_channels, 2, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)  # Feature extraction
        depth_params = self.depth_head(features)  # [B, 2, H/32, W/32]

        # Split into depth mean and log variance
        depth_mean_raw = depth_params[:, 0:1, :, :]
        log_var_raw = depth_params[:, 1:2, :, :]

        # Normalize mean prediction between min_depth and max_depth
        depth_mean = torch.sigmoid(depth_mean_raw) * (self.max_depth - self.min_depth) + self.min_depth

        # Ensure variance is positive
        depth_var = torch.exp(log_var_raw)

        # Upsample to match input size
        depth_mean = F.interpolate(depth_mean, size=x.shape[2:], mode='bilinear', align_corners=False)
        depth_var = F.interpolate(depth_var, size=x.shape[2:], mode='bilinear', align_corners=False)

        return depth_mean, depth_var
