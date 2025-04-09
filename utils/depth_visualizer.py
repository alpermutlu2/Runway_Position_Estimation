
import numpy as np
import cv2
import torch

def colorize_depth(depth_tensor, min_depth=0.1, max_depth=100.0, colormap=cv2.COLORMAP_INFERNO):
    """
    Converts a depth tensor to a colorized OpenCV image for visualization.

    Args:
        depth_tensor (torch.Tensor): Depth map tensor (1, H, W) or (H, W)
        min_depth (float): Minimum depth for normalization
        max_depth (float): Maximum depth for normalization
        colormap (int): OpenCV colormap type

    Returns:
        np.ndarray: Colorized depth image in uint8 format
    """
    if isinstance(depth_tensor, torch.Tensor):
        depth_tensor = depth_tensor.squeeze().cpu().numpy()

    depth_clipped = np.clip(depth_tensor, min_depth, max_depth)
    depth_norm = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_img = (depth_norm * 255).astype(np.uint8)

    color_depth = cv2.applyColorMap(depth_img, colormap)
    return color_depth

def overlay_images(image, depth_color, alpha=0.5):
    """
    Overlays colorized depth map on top of RGB image.

    Args:
        image (np.ndarray): RGB image (H, W, 3)
        depth_color (np.ndarray): Colorized depth image (H, W, 3)
        alpha (float): Blending factor

    Returns:
        np.ndarray: Overlayed image
    """
    return cv2.addWeighted(image, 1 - alpha, depth_color, alpha, 0)
