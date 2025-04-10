
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Normalize

class MiDaSDepth:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_type = model_type
        if model_type == "DPT_Large":
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device)
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        else:
            raise NotImplementedError(f"Model type {model_type} not supported.")

        self.model.eval()

    def predict(self, image):
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # normalize
        return depth_map
