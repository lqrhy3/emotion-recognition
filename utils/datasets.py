import torch
from torch.utils.data import Dataset
from utils import utils
from utils import transforms
import os
import cv2
import numpy as np
import json


class DetectionDataset(Dataset):
    """Face detection dataset.
    Making custom dataset for detection task. Overloading __getitem__ and __len__ methods
    for torch.Dataloader compatibility.
    """
    dir_ = os.path.dirname(__file__)

    def __init__(self, grid_size, num_bboxes, path='../data/detection/', transform=None):
        self.datadir = os.path.join(self.dir_, path, 'train_images')
        self.img_names = np.array(os.listdir(self.datadir))
        self.markup_dir = os.path.join(self.dir_, path, 'data_markup.txt')
        self.transform = transform

        self.S = grid_size
        self.B = num_bboxes

        with open(self.markup_dir, 'r') as file:
            self.markup = json.load(file)

    def __getitem__(self, idx):
        """Converting .jpg image to torch format, passing it through transformations, modifying bbox respectively and
        conctructing YOLO output tensor from it.
        Returns:
            (Tensor) Transformed image. Sized (3 x image_w x image_h)
            (Tensor) YOLO format output. Sized (5*num_bboxes*num_classes x grid_size x grid_size)
            (Tensor) Face rectangular coordinates. [x_lt, y_lt, x_rb, y_rb]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.datadir, self.img_names[idx])
        img = cv2.imread(img_name)

        face_rect = np.array(self.markup[self.img_names[idx][:16]])

        if self.transform:
            sample = self.transform(image=img, bboxes=[utils.xywh2xyxy(face_rect)], labels=['face'])
            #img, face_rect = sample['image'], utils.xyxy2xywh(sample['bboxes'][0])
            img, face_rect = sample['image'], sample['bboxes'][0]

        #target = utils.to_yolo_target(face_rect, img.shape[0], self.S, self.B)
        target = utils.to_yolo_target(utils.xyxy2xywh(face_rect), img.shape[0], self.S, self.B)
        img = transforms.ImageToTensor()(img)

        return img, target, torch.tensor(face_rect)

    def __len__(self):
        return len(self.img_names)
