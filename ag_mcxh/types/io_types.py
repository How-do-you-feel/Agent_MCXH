import numpy as np
from PIL import Image

class ImageIO:
    def __init__(self, image_path):
        self.image_path = image_path
        self._image = None

    def to_array(self):
        if self._image is None:
            self._image = np.array(Image.open(self.image_path))
        return self._image

    def to_pil(self):
        return Image.open(self.image_path)