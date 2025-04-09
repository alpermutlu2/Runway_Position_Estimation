
import torch

def uncertainty_loss(depth_mean, depth_var, gt_depth, mask=None, eps=1e-6):
    """
    Aleatoric uncertainty-aware loss function.

    Args:
        depth_mean (torch.Tensor): Predicted depth mean (B, 1, H, W)
        depth_var (torch.Tensor): Predicted depth variance (B, 1, H, W)
        gt_depth (torch.Tensor): Ground truth depth (B, 1, H, W)
        mask (torch.Tensor, optional): Boolean mask of shape (B, 1, H, W) or (B, H, W)
        eps (float): Small constant to prevent log(0)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Safety checks
    assert depth_mean.shape == gt_depth.shape == depth_var.shape,         "All input tensors must have the same shape"

    # Create default mask if not provided
    if mask is None:
        mask = gt_depth > 0
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # Ensure all tensors are on the same device
    mask = mask.to(depth_mean.device)

    # Compute per-pixel negative log likelihood of Gaussian
    sigma = depth_var[mask] + eps
    residual = (depth_mean[mask] - gt_depth[mask]) ** 2
    loss = 0.5 * (torch.log(sigma) + residual / sigma)

    return loss.mean()
