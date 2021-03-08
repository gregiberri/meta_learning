import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class rotate_image(object):

    def __init__(self, k, channels):
        self.k = k
        self.channels = channels

    def __call__(self, image):
        if self.channels == 1 and len(image.shape) == 3:
            image = image[:, :, 0]
            image = np.expand_dims(image, axis=2)

        elif self.channels == 1 and len(image.shape) == 4:
            image = image[:, :, :, 0]
            image = np.expand_dims(image, axis=3)

        image = np.rot90(image, k=self.k).copy()
        return image


class torch_rotate_image(object):

    def __init__(self, k, channels):
        self.k = k
        self.channels = channels

    def __call__(self, image):
        rotate = transforms.RandomRotation(degrees=self.k * 90)
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        image = Image.fromarray(image)
        image = rotate(image)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image
