import torch
from yolo_model import TinyYolo
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader
import numpy as np
# from train_yolo import PATH_TO_SAVE
import os
import albumentations
from utils.utils import get_object_cell, xywh2xyxy, from_yolo_target
from data.detection.show_targets import show_rectangles

model = TinyYolo()
load = torch.load(os.path.join('log', 'checkpoint_test.pt'))
model.load_state_dict(load['model_state_dict'])

model.eval()

train_transforms = albumentations.Compose([
    albumentations.Resize(height=384, width=384)
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=train_transforms)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

for image, target in dataloader:
    output = model(image)
    idx = get_object_cell(target)
    # yolo_output = from_yolo_target(output[:, :10, :, :], image.size(2))[(idx[0]*6 + idx[1])*2:(idx[0]*6 + idx[1])*2 + 2, :]
    listed_output = from_yolo_target(output[:, :10, :, :], image.size(2))
    pred_output = np.expand_dims(listed_output[np.argmax(listed_output[:, 4]).item(), :], axis=0)
    show_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)), np.expand_dims(xywh2xyxy(pred_output[:, :4]), axis=0), pred_output[:, 4])
