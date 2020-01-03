import torch
from tiny_yolo_model import TinyYolo
from utils.logger import Logger
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import albumentations
from utils.loss import Loss
from collections import Counter
from utils.utils import from_yolo_target, xywh2xyxy
import os
import datetime
import numpy as np


# Declaring constants for logging and creating dir for current experiment. Initiating logger.
TASK = 'detection'  # detection or emorec
TEST = False
PATH_TO_LOG = 'log/' + TASK
SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
COMMENT = 'new trainer test'

if not TEST:
    os.mkdir(os.path.join(PATH_TO_LOG, SESSION_ID))
    logger = Logger('logger', task=TASK, session_id=SESSION_ID)

# Declaring hyperparameters
n_epoch = 20
batch_size = 14
grid_size = 7
num_bboxes = 2
val_split = 0.03

# Initiating model and device (cuda/cpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TinyYolo(grid_size=grid_size, num_bboxes=num_bboxes).to(device)

# Initiating optimizer and scheduler for training steps
# optim = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
optim = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [10], gamma=1.0)

# Declaring augmentations for images and bboxes
train_transforms = albumentations.Compose([
    albumentations.RandomSizedBBoxSafeCrop(height=448, width=448, always_apply=True),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Rotate(15, p=0.5),

], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

# Decalring dataset and creating dataloader for train and validation phase
dataset = DetectionDataset(transform=train_transforms, grid_size=grid_size, num_bboxes=num_bboxes)

dataset_len = len(dataset)
val_len = int(np.floor(val_split * dataset_len))
val_idxs = np.random.choice(list(range(dataset_len)), val_len)
train_idxs = list(set(list(range(dataset_len))) - set(val_idxs))

train_sampler = SubsetRandomSampler(train_idxs)
val_sampler = SubsetRandomSampler(val_idxs)

train_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, sampler=train_sampler)
val_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, sampler=val_sampler)

# Declaring loss function
loss = Loss(grid_size=grid_size, num_bboxes=num_bboxes)


if not TEST:
    logger.start_info(optim=optim, hyperparameters={'n_epoch': n_epoch, 'batch_size': batch_size,
                                                    'grid_size': grid_size, 'num_bboxes': num_bboxes},
                      transforms=train_transforms, comment=COMMENT)

# Training loop
for epoch in range(n_epoch):

    for phase in ['train', 'val']:
        epoch_loss = Counter()
        iou = 0

        if phase == 'train':
            dataloader = train_dataloader
            model.train()
        else:
            dataloader = val_dataloader
            model.eval()

        for i, (image, target, face_rect) in enumerate(dataloader):
            image = image.to(device)
            target = target.to(device)

            optim.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                output = model(image)
                loss_value, logger_loss = loss(output, target)
                if phase == 'train':
                    # Parameters updating
                    loss_value.backward()
                    optim.step()
                    scheduler.step(epoch)
                else:
                    # Computing metrics at validation phase
                    face_rect.to(device)
                    listed_output = from_yolo_target(output, image_w=448, grid_size=grid_size, num_bboxes=num_bboxes)
                    semi_pred = np.expand_dims(listed_output[np.argmax(listed_output[:, 4]).item(), :], axis=0)
                    iou += loss._compute_iou(face_rect,
                                             torch.tensor(np.expand_dims(xywh2xyxy(semi_pred[:, :4]), axis=0)).to(device))

            epoch_loss += logger_loss

        epoch_loss = Counter({key: value / (i + 1) for key, value in epoch_loss.items()})
        iou = iou / (i + 1)
        if not TEST:
            # Logging
            logger.epoch_info(epoch=epoch, loss=epoch_loss, val_metrics=iou, phase=phase)

            if phase == 'train' and epoch % 5 == 0:
                # Checkpoint. Saving model, optimizer, scheduler and train info
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_value.item()
                }, os.path.join(PATH_TO_LOG, SESSION_ID, 'checkpoint.pt'))
                logger.logger.info('!Checkpoint created!')
