import torch

def uncertainty_loss(depth_mean, depth_var, gt_depth, mask=None):
    """
    Args:
        depth_mean: Predicted depth (B, 1, H, W)
        depth_var: Predicted variance (B, 1, H, W)
        gt_depth: Ground truth depth (B, 1, H, W)
        mask: Optional mask to ignore regions (e.g., dynamic objects)
    """
    if mask is None:
        mask = (gt_depth > 0)  # Default: valid depth pixels
    
    sigma = depth_var[mask] + 1e-6
    residual = (depth_mean[mask] - gt_depth[mask]) ** 2
    return 0.5 * (torch.log(sigma) + residual / sigma).mean()