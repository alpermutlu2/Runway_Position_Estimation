
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SimpleDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.image_dir = image_dir
        self.depth_dir = depth_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_dir, self.image_files[idx]))
        depth = np.load(os.path.join(self.depth_dir, self.image_files[idx].replace('img', 'depth')))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        depth = torch.tensor(depth).unsqueeze(0).float()
        return image, depth


class DummyDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


def train_model(image_path, depth_path, epochs=3, batch_size=4, lr=1e-3):
    dataset = SimpleDepthDataset(image_path, depth_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DummyDepthModel()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, depths in dataloader:
            preds = model(images)
            loss = criterion(preds, depths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/depth_model.pth")
    print("Training complete. Model saved to checkpoints/depth_model.pth")


if __name__ == "__main__":
    train_model("data/images/train", "data/depth_gt/train", epochs=5)
