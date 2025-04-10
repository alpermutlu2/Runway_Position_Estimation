# tools/export_segmentation_onnx.py

import torch
import torchvision.models.segmentation as models

def export_deeplab_onnx(output_path="semantic_models/deeplabv3.onnx"):
    model = models.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model,
                      dummy_input,
                      output_path,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=11,
                      dynamic_axes={"input": {0: "batch_size"},
                                    "output": {0: "batch_size"}})

    print(f"ONNX model exported to {output_path}")

if __name__ == "__main__":
    export_deeplab_onnx()
