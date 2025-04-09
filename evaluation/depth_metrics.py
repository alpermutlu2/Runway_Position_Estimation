
import torch
import numpy as np

def compute_depth_metrics(pred, gt, mask=None, eps=1e-6):
    """
    Computes standard depth estimation metrics.

    Args:
        pred (torch.Tensor): Predicted depth map (B, 1, H, W)
        gt (torch.Tensor): Ground truth depth map (B, 1, H, W)
        mask (torch.Tensor, optional): Boolean mask (B, 1, H, W) or (B, H, W)
        eps (float): Small value to avoid division by zero

    Returns:
        dict: Dictionary of depth evaluation metrics
    """
    assert pred.shape == gt.shape, "Prediction and GT must have same shape"
    if mask is None:
        mask = gt > 0
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(pred.device)

    pred = pred[mask]
    gt = gt[mask]

    pred = torch.clamp(pred, min=eps)
    gt = torch.clamp(gt, min=eps)

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    ratio = torch.max(gt / pred, pred / gt)
    a1 = (ratio < 1.25).float().mean()
    a2 = (ratio < 1.25 ** 2).float().mean()
    a3 = (ratio < 1.25 ** 3).float().mean()

    return {
        "abs_rel": abs_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "a1": a1.item(),
        "a2": a2.item(),
        "a3": a3.item(),
    }
