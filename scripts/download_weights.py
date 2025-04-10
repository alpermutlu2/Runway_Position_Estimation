
# File: scripts/download_weights.py

import os
import urllib.request
from pathlib import Path

weights = {
    "models/midas/dpt_large_384.pt": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt",
    "models/raft/raft-sintel.pth": "https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-sintel.pth",
    "models/raft/raft-things.pth": "https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth",
    "models/panoptic/model_final_cafdb1.pkl": "https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/137849600/model_final_cafdb1.pkl",
    "models/panoptic/config.yaml": "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
}

for local_path, url in weights.items():
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        print(f"Downloading {url} → {local_path}")
        urllib.request.urlretrieve(url, local_path)
    else:
        print(f"✅ Already exists: {local_path}")

print("\n🎉 All required weights are now downloaded.")
