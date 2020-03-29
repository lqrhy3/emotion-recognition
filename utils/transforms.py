import torch
import albumentations
from PIL import Image
import numpy as np


class ImageToTensor:
    def __call__(self, img):
        # opencv image: W x H x C
        # torch image: C x W x H
        image = img.transpose((2, 0, 1))

        return torch.from_numpy(image)


class GaussNoise:
    def __init__(self, *args, **kwargs):
        self.transform = albumentations.GaussNoise(*args, **kwargs)

    def __call__(self, img):
        img = self.transform(image=np.array(img))['image']
        return Image.fromarray(img)

    def __repr__(self):
        return self.transform.__repr__()
