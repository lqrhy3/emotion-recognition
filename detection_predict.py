import torch
from yolo_model import TinyYolo
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader
# from train_yolo import PATH_TO_SAVE
import os
import albumentations
from utils.utils import get_object_cell, xywh2xyxy, from_yolo_target
from data.detection.show_targets import show_rectangles

model = TinyYolo()
load = torch.load(os.path.join('log', 'checkpoint.pt'))
model.load_state_dict(load['model_state_dict'])

model.eval()

train_transforms = albumentations.Compose([
    albumentations.Resize(height=384, width=384)
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=train_transforms)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

image, target = next(iter(dataloader))
output = model(image)
idx = get_object_cell(target)
show_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)), xywh2xyxy(from_yolo_target(output[:, :10, :, :],
                image.size(2))[(idx[0]*6 + idx[1])*2:(idx[0]*6 + idx[1])*2 + 2, :4]))
