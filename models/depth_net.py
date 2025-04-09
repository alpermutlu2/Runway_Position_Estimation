import torch
import torch.nn as nn
from torchvision.models import resnet18

class ProbabilisticDepthNet(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        # Backbone initialization
        if backbone == "resnet18":
            self.encoder = resnet18(pretrained=True)
            # Remove avgpool and fc layers, keep conv layers
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.encoder_out_channels = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")
        
        # Depth head (mean + variance)
        self.depth_head = nn.Conv2d(self.encoder_out_channels, 2, kernel_size=1)
        
        # Depth scaling (optional, adjust based on dataset)
        self.min_depth = 0.1
        self.max_depth = 100.0

    def forward(self, x):
        features = self.encoder(x)
        depth_params = self.depth_head(features)
        
        # Upsample to input resolution (if needed)
        depth_params = nn.functional.interpolate(
            depth_params, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Depth mean (sigmoid -> scale to [min_depth, max_depth])
        depth_mean = torch.sigmoid(depth_params[:, 0:1]) * (self.max_depth - self.min_depth) + self.min_depth
        
        # Variance (softplus to ensure positivity)
        depth_var = nn.functional.softplus(depth_params[:, 1:2])
        
        return depth_mean, depth_var