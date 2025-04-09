
import torch
import numpy as np
import random
from models.depth_net import ProbabilisticDepthNet
from models.sam_integration import SAMDynamicMasker
from losses.uncertainty_loss import uncertainty_loss
from datasets.kitti_loader import KITTIDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from depth_metrics import compute_depth_metrics
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    metric_list = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            gt_depth = batch["depth"].to(device)

            depth_mean, depth_var = model(image)
            loss = uncertainty_loss(depth_mean, depth_var, gt_depth)
            total_loss += loss.item()

            metrics = compute_depth_metrics(depth_mean, gt_depth)
            metric_list.append(metrics)

    avg_metrics = {k: np.mean([m[k] for m in metric_list]) for k in metric_list[0].keys()}
    return total_loss / len(dataloader), avg_metrics

def train(config):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and SAM
    model = ProbabilisticDepthNet(backbone=config["backbone"]).to(device)
    sam = SAMDynamicMasker()

    # Dataset
    dataset = KITTIDataset(config["data_path"])
    val_split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            gt_depth = batch["depth"].to(device)

            # Batch-wise SAM masking
            masks = []
            for img in image:
                mask_np = sam.generate_mask(img.cpu().numpy().transpose(1, 2, 0))
                masks.append(torch.from_numpy(mask_np).unsqueeze(0))
            sam_mask = torch.cat(masks, dim=0).to(device)

            # Forward pass
            depth_mean, depth_var = model(image)
            loss = uncertainty_loss(depth_mean, depth_var, gt_depth, mask=~sam_mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{config['epochs']}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"AbsRel: {val_metrics['abs_rel']:.4f}, "
              f"RMSE: {val_metrics['rmse']:.4f}, "
              f"RMSE_log: {val_metrics['rmse_log']:.4f}, "
              f"A1: {val_metrics['a1']:.4f}, "
              f"A2: {val_metrics['a2']:.4f}, "
              f"A3: {val_metrics['a3']:.4f}")

        torch.save(model.state_dict(), f"checkpoints/depth_model_epoch{epoch+1}.pth")
