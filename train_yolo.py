import torch
from yolo_model import TinyYolo
from utils.logger import Logger
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader
import albumentations
from utils.loss import Loss
"""Initiate logger"""
logger = Logger('Yolo')


"""Initiate model"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TinyYolo().to(device)


"""Hyperparameters"""
n_epoch = 10
grid_size = 6
num_bboxes = 2

"""Initiate optimizers"""
optim = torch.optim.Adam(net.parameters())


train_transforms = albumentations.Compose([
    albumentations.Resize(height=384, width=384)
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=train_transforms)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
loss = Loss(grid_size=grid_size, num_bboxes=num_bboxes)

logger.start_info(optim=optim)
for epoch in range(n_epoch):
    loss_value = None
    for image, target in dataloader:
        optim.zero_grad()
        output = net(image)

        loss_value = loss(output, target)
        loss_value.backward()
        optim.step()

        logger.epoch_info(epoch=epoch, train_loss=loss_value.item())

