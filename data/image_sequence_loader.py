
import os
import cv2

class ImageSequenceLoader:
    def __init__(self, folder, resize=(640, 480)):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
        self.index = 0
        self.resize = resize

    def __len__(self):
        return len(self.files)

    def get_next(self):
        if self.index >= len(self.files):
            return None, None
        file = self.files[self.index]
        path = os.path.join(self.folder, file)
        image = cv2.imread(path)
        if self.resize:
            image = cv2.resize(image, self.resize)
        timestamp = self.index / 30.0
        self.index += 1
        return image, timestamp
