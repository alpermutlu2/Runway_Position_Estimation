
class ModelWrapper:
    def __init__(self, name):
        self.name = name
        if name == 'M4Depth':
            self.model = self.load_m4depth()
        elif name == 'CoDEPS':
            self.model = self.load_codeps()
        else:
            raise ValueError(f'Unknown model name: {name}')

    def load_m4depth(self):
        # Placeholder for loading actual M4Depth model
        return lambda image: image.mean(axis=2)  # dummy grayscale depth

    def load_codeps(self):
        # Placeholder for loading actual CoDEPS model
        return lambda image_pair: image_pair[0].mean(axis=2)  # dummy grayscale

    def predict(self, *args):
        return self.model(*args)
