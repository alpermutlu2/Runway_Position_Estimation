
import os
import shutil
import random

def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:]
    }

    for split_name, split_files in split.items():
        for f in split_files:
            shutil.copy2(os.path.join(image_dir, f), os.path.join(output_dir, split_name, f))

if __name__ == '__main__':
    split_dataset('data/images/train', 'data/images', 0.7, 0.15)
    split_dataset('data/depth_gt/train', 'data/depth_gt', 0.7, 0.15)
    split_dataset('data/poses_gt/train', 'data/poses_gt', 0.7, 0.15)
