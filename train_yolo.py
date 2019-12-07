import torch
from yolo_model import TinyYolo
from utils.logger import Logger
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader
import albumentations
from utils.loss import Loss
from collections import Counter
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
n_epoch = 101
batch_size = 32
grid_size = 6
num_bboxes = 2

"""Initiate optimizers"""
optim = torch.optim.SGD(model.parameters(), lr=0.00003, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [7, 15, 25], gamma=0.3)

train_transforms = albumentations.Compose([
    albumentations.RandomSizedBBoxSafeCrop(height=384, width=384, always_apply=True),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Rotate(15, p=0.5),

], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=train_transforms)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
loss = Loss(grid_size=grid_size, num_bboxes=num_bboxes)

logger.start_info(optim=optim, hyperparameters={'n_epoch': n_epoch, 'batch_size': batch_size,
                                                'grid_size': grid_size, 'num_bboxes': num_bboxes},
                  transforms=train_transforms, comment='TinyYOLO')

for epoch in range(n_epoch):
    epoch_loss = Counter()

    for image, target in dataloader:
        # show_rectangles(image=image.numpy().squeeze(0).transpose((1, 2, 0)), rectangles=xywh2xyxy(from_yolo_target(
        #     target[:, :10, :, :], image_w=image.size(2))))
        image = image.to(device)
        target = target.to(device)

        optim.zero_grad()
        output = model(image)

        loss_value, logger_loss = loss(output, target)
        loss_value.backward()
        optim.step()
        epoch_loss += logger_loss

    logger_loss = Counter({key: value / batch_size for key, value in logger_loss.items()})
    logger.epoch_info(epoch=epoch, train_loss=logger_loss)

    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': loss_value.item()
        }, os.path.join(PATH_TO_SAVE, 'checkpoint.pt'))
        logger.logger.info('!Checkpoint created!')

