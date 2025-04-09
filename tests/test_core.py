
import torch
import pytest
from models.depth_net import ProbabilisticDepthNet
from losses.uncertainty_loss import uncertainty_loss
from evaluation.depth_metrics import compute_depth_metrics

def test_model_forward():
    model = ProbabilisticDepthNet().eval()
    x = torch.randn(1, 3, 128, 416)
    with torch.no_grad():
        mean, var = model(x)
    assert mean.shape == x[:, :1].shape
    assert var.shape == x[:, :1].shape
    assert torch.all(var > 0)

def test_uncertainty_loss():
    pred = torch.ones(1, 1, 64, 64)
    var = torch.ones(1, 1, 64, 64) * 0.1
    gt = torch.ones(1, 1, 64, 64)
    loss = uncertainty_loss(pred, var, gt)
    assert isinstance(loss.item(), float)
    assert loss.item() < 1e-3

def test_depth_metrics():
    gt = torch.ones(1, 1, 64, 64)
    pred = gt * 0.9
    metrics = compute_depth_metrics(pred, gt)
    assert 'abs_rel' in metrics and 'a1' in metrics
    assert 0 <= metrics['abs_rel'] < 1
    assert 0 <= metrics['a1'] <= 1
