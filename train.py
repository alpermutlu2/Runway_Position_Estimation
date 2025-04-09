import torch
from models.depth_net import ProbabilisticDepthNet
from models.sam_integration import SAMDynamicMasker
from losses.uncertainty_loss import uncertainty_loss
from datasets.kitti_loader import KITTIDataset

def train(config):
    # Initialize model and SAM
    model = ProbabilisticDepthNet(backbone=config["backbone"]).cuda()
    sam = SAMDynamicMasker()
    
    # Dataset
    dataset = KITTIDataset(config["data_path"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    for epoch in range(config["epochs"]):
        for batch in dataloader:
            image = batch["image"].cuda()
            gt_depth = batch["depth"].cuda()
            
            # Generate SAM mask for dynamic objects
            with torch.no_grad():
                sam_mask = sam.generate_mask(image[0].cpu().numpy().transpose(1,2,0))
                sam_mask = torch.from_numpy(sam_mask).cuda().unsqueeze(0)
            
            # Forward pass
            depth_mean, depth_var = model(image)
            
            # Mask out dynamic regions
            loss = uncertainty_loss(depth_mean, depth_var, gt_depth, mask=~sam_mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()