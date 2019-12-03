import torch
from yolo_model import TinyYolo
from utils.logger import Logger
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader
import albumentations
from utils.loss import Loss
from data.detection.show_targets import show_rectangles
from utils.utils import from_yolo_target, xywh2xyxy
from utils.utils import get_object_cell
import os

"""Initiate logger"""
PATH_TO_SAVE = 'log'
logger = Logger('Yolo')


"""Initiate model"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TinyYolo().to(device)
model.train()


"""Hyperparameters"""
n_epoch = 10
batch_size = 1
grid_size = 6
num_bboxes = 2

"""Initiate optimizers"""
optim = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0005)


train_transforms = albumentations.Compose([
    albumentations.Resize(height=384, width=384)
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=train_transforms)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
loss = Loss(grid_size=grid_size, num_bboxes=num_bboxes)

logger.start_info(optim=optim, hyperparameters={'n_epoch': n_epoch, 'batch_size': batch_size,
                                                'grid_size': grid_size, 'num_bboxes': num_bboxes},
                  transforms=train_transforms)

for epoch in range(n_epoch):
    loss_value = torch.tensor([float('inf')])

    if epoch % 1 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': loss_value.item()
        }, os.path.join(PATH_TO_SAVE, 'checkpoint.pt'))
        logger.logger.info('!Checkpoint created!')

    for image, target in dataloader:
        optim.zero_grad()
        output = model(image)

        loss_value, logger_loss = loss(output, target)
        loss_value.backward()
        optim.step()
        logger.epoch_info(epoch=epoch, train_loss=logger_loss)
        # idx = get_object_cell(target)
        # show_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)), xywh2xyxy(from_yolo_target(output[:, :10, :, :],
        #                         image.size(2))[(idx[0]*grid_size + idx[1])*2:(idx[0]*grid_size + idx[1])*2 + 2, :4]))

