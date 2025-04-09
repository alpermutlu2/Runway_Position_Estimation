
import argparse
import os
import cv2
import torch
from models.depth_net import ProbabilisticDepthNet
from datasets.kitti_loader import KITTIDataset
from torch.utils.data import DataLoader
from utils.depth_visualizer import colorize_depth, overlay_images
from torchvision.transforms.functional import to_pil_image

def main():
    parser = argparse.ArgumentParser(description="Visualize predicted depth maps")
    parser.add_argument('--data_path', type=str, required=True, help='Dataset root path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--out_dir', type=str, default="outputs/viz", help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--backbone', type=str, default='resnet18')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = ProbabilisticDepthNet(backbone=args.backbone).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # Load dataset
    dataset = KITTIDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            image = batch["image"].cuda()
            gt_depth = batch["depth"]

            pred_depth, _ = model(image)
            pred_depth = pred_depth.squeeze(0)

            # Visualization
            rgb_np = (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            depth_color = colorize_depth(pred_depth)
            overlay = overlay_images(rgb_np, depth_color)

            cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}_depth.png"), depth_color)
            cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}_overlay.png"), overlay)
            to_pil_image(gt_depth[0]).save(os.path.join(args.out_dir, f"{i:03d}_gt_depth.png"))

            print(f"[{i+1}/{args.num_samples}] Saved visualizations.")

if __name__ == "__main__":
    main()
