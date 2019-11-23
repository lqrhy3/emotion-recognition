import torch
from yolo_model import TinyYolo
from utils.logger import Logger
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader
from data.detection import show_targets
import albumentations
import cv2
"""Initiate logger"""
logger = Logger('Yolo')


"""Initiate model"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TinyYolo().to(device)

train_transforms = albumentations.Compose([
    # albumentations.RandomCrop(height=384, width=384, )
    albumentations.Resize(height=384, width=384)
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

test_transforms = albumentations.Compose([
])

dataset = DetectionDataset(transforms=train_transforms)
dataloader = DataLoader(dataset, shuffle=False






















                        )

for img, rect in dataloader:
    rect = [int(x.item()) for x in rect]
    print(rect)
    img = img.numpy().squeeze(0)
    show_targets.show_rectangle(img, rect)





