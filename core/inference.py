
import numpy as np
from models.model_wrapper import ModelWrapper

def run_inference_pipeline(input_path, model_names):
    # Dummy input (replace with real data loading)
    image = np.random.rand(128, 128, 3)
    image_pair = (image, image)

    depth_sources = []
    confidence_maps = []
    for model_name in model_names:
        model = ModelWrapper(model_name)
        if model_name == 'M4Depth':
            depth = model.predict(image)
        elif model_name == 'CoDEPS':
            depth = model.predict(image_pair)
        confidence = np.ones_like(depth)
        depth_sources.append(depth)
        confidence_maps.append(confidence)

    outputs = {
        'depth_sources': depth_sources,
        'confidence_maps': confidence_maps,
        'depth': depth_sources[0],
        'pose': [np.eye(3, 4)] * 10,
        'flow': [np.random.rand(128, 128, 2)] * 9,
        'landmarks': [np.random.rand(3) for _ in range(10)],
        'observations': {(i, i): np.array([32, 32]) for i in range(10)}
    }
    return outputs
