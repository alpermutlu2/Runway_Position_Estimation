
# File: scripts/check_weights.py

from pathlib import Path

REQUIRED_FILES = [
    "models/midas/dpt_large_384.pt",
    "models/m4depth/m4depth_model.ckpt",
    "models/raft/raft-sintel.pth",
    "models/raft/raft-things.pth",
    "models/panoptic/model_final_cafdb1.pkl",
    "models/panoptic/config.yaml",
    "models/onnx/midas.onnx",
    "models/onnx/superpoint.onnx",
    "models/onnx/lightglue.onnx",
    "models/onnx/raft.onnx"
]

def check_weights(verbose=True):
    missing = []
    for path in REQUIRED_FILES:
        if not Path(path).is_file():
            missing.append(path)

    if missing and verbose:
        print("\n🚫 Missing model files:")
        for m in missing:
            print(f" - {m}")
        print("\nPlease download missing files using scripts/download_weights.py or check ONNX exports.")
        return False
    elif not missing and verbose:
        print("✅ All required model weights are present!")
    return len(missing) == 0

if __name__ == "__main__":
    check_weights()
