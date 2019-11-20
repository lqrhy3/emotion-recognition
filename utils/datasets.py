import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import json


class DetectionDataset(Dataset):
    """Face detection dataset"""
    dir_ = os.path.dirname(__file__)

    def __init__(self, path='../data/detection/'):
        self.datadir = os.path.join(self.dir_, path, 'images')
        self.img_names = np.array(os.listdir(self.datadir))
        self.markup_dir = os.path.join(self.dir_, path, 'data_markup.txt')

        with open(self.markup_dir, 'r') as file:
            self.markup = json.load(file)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = os.path.join(self.datadir, self.img_names[item])
        img = cv2.imread(img_name)

        face_rect = self.markup[self.img_names[item][:16]]
        sample = (img, face_rect)

        return sample

    def __len__(self):
        return len(self.img_names)


