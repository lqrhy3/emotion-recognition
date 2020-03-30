import albumentations
import datetime
import numpy as np
import os
import torch
from models.detection.faced_model import FacedModel

from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.datasets import DetectionDataset
from utils.logger import Logger
from utils.loss import Loss, LossCounter
from utils.utils import from_yolo_target, xywh2xyxy, compute_iou


# Declaring constants for logging and creating dir for current experiment. Initiating logger.
TASK = 'detection'  # detection or emorec
TEST = False
PATH_TO_LOG = 'log/' + TASK
SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
COMMENT = 'FacedModel, 9x9 grid, QuantStub enabled. light version'

if not TEST:
    os.makedirs(os.path.join(PATH_TO_LOG, SESSION_ID), exist_ok=True)
    logger = Logger('logger', task=TASK, session_id=SESSION_ID)

# Declaring hyperparameters
n_epoch = 130
batch_size = 26
image_size = (288, 288)
grid_size = 9
num_bboxes = 2
val_split = 0.03

# Initiating detection_model and device (cuda/cpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = TinyYolo(grid_size=grid_size, num_bboxes=num_bboxes).to(device)
model = FacedModel(grid_size=grid_size, num_bboxes=num_bboxes).to(device)

# Initiating optimizer and scheduler for training steps
optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [1], gamma=1)

# Declaring augmentations for images and bboxes
train_transforms = albumentations.Compose([
    albumentations.RandomSizedBBoxSafeCrop(height=image_size[0], width=image_size[1], always_apply=True),
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
    logger.start_info(optim=optim, scheduler=scheduler, hyperparameters={'n_epoch': n_epoch, 'batch_size': batch_size,
                                                                         'grid_size': grid_size, 'num_bboxes': num_bboxes},
                      transforms=train_transforms, comment=COMMENT)

torch.save(model, os.path.join(PATH_TO_LOG, SESSION_ID, 'model.pt'))
# Training loop
for epoch in range(n_epoch):
    batch_train_loss = LossCounter()
    batch_val_loss = LossCounter()
    batch_val_metrics = 0

    for phase in ['train', 'val']:

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

                    batch_train_loss += logger_loss
                else:
                    # Computing metrics at validation phase

                    face_rect.to(device)
                    listed_output = torch.tensor(from_yolo_target(output, image_w=image_size[0], grid_size=grid_size, num_bboxes=num_bboxes))
                    preds = torch.empty((listed_output.size(0), 5))
                    idxs = torch.argmax(listed_output[:, :, 4], dim=1)
                    for batch in range(listed_output.size(0)):
                        preds[batch] = listed_output[batch, idxs[batch], ...]

                    batch_val_loss += logger_loss
                    batch_val_metrics += compute_iou(face_rect,
                                                     torch.tensor(xywh2xyxy(preds[:, :4]), dtype=torch.float), num_bboxes=2).mean().item()
    epoch_train_loss = LossCounter({key: value / len(train_dataloader) for key, value in batch_train_loss.items()})
    epoch_val_loss = LossCounter({key: value / len(val_dataloader) for key, value in batch_val_loss.items()})
    epoch_val_metrics = batch_val_metrics / len(val_dataloader)
    if not TEST:
        # Logging
        logger.epoch_info(epoch=epoch, train_loss=epoch_train_loss, val_loss=epoch_val_loss, val_metrics=epoch_val_metrics)

    if epoch % 5 == 0:
        # Checkpoint. Saving detection_model, optimizer, scheduler and train info
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': batch_train_loss['Total loss']
        }, os.path.join(PATH_TO_LOG, SESSION_ID, 'checkpoint.pt'))
        logger.info('!Checkpoint created!')
