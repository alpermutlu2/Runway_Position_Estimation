# inference/segmentation.py

import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
import numpy as np
import cv2


class RunwaySegmenter:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.eval().to(self.device)

        # Placeholder: Define the class index for 'runway' (in fine-tuned version)
        self.runway_class_index = 15  # example only

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            pred_mask = output.argmax(0).cpu().numpy()

        # Convert to binary runway mask
        binary_mask = (pred_mask == self.runway_class_index).astype(np.uint8) * 255
        binary_mask = cv2.resize(binary_mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return binary_mask
