import torch


class ImageToTensor(object):
    def __call__(self, img):
        # opencv image: W x H x C
        # torch image: C x W x H
        image = img.transpose((2, 0, 1))

        return torch.from_numpy(image)
