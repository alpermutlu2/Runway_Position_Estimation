
import os
import torch
import numpy as np
import pytest
from types import SimpleNamespace
from models.sam_integration import SAMDynamicMasker
from train_cli import main as train_cli_main

@pytest.mark.skipif(not os.path.exists("sam_vit_h.pth"), reason="SAM model checkpoint not available")
def test_sam_mask_shape():
    masker = SAMDynamicMasker(checkpoint_path="sam_vit_h.pth")
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = masker.generate_mask(image)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.bool_

def test_cli_config_parsing(monkeypatch):
    called = {}
    def mock_train(config):
        called["config"] = config

    monkeypatch.setattr("train.train", mock_train)

    args = [
        "train_cli.py",
        "--data_path", "./dummy",
        "--batch_size", "2",
        "--epochs", "1",
        "--lr", "0.001",
        "--backbone", "resnet18"
    ]

    monkeypatch.setattr("sys.argv", args)
    train_cli_main()
    assert called["config"]["data_path"] == "./dummy"
    assert called["config"]["batch_size"] == 2
