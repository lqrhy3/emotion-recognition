import albumentations
import argparse
import datetime
import numpy as np
import os
import torch

import sys
module_path = os.path.abspath(os.getcwd() + '\\..')
if module_path not in sys.path:
    sys.path.append(module_path)

from models.detection.tiny_yolo_model import TinyYolo
from models.detection.faced_model import FacedModel, FacedModelLite
from utils.datasets import DetectionDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.logger import Logger
from utils.loss import Loss, LossCounter
from utils.utils import from_yolo_target, xywh2xyxy, compute_iou


def run_train():
    if not TEST:
        os.makedirs(os.path.join(PATH_TO_LOG, SESSION_ID), exist_ok=True)
        logger = Logger('logger', task=TASK, session_id=SESSION_ID)

    # Initiating detection_model and device (cuda/cpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = net(grid_size=grid_size, num_bboxes=num_bboxes).to(device)

    # Initiating optimizer and scheduler for training steps
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1], gamma=1)

    # Declaring augmentations for images and bboxes
    train_transform = albumentations.Compose([
        albumentations.RandomSizedBBoxSafeCrop(height=image_size[0], width=image_size[1], always_apply=True),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(15, p=0.5),

    ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Decalring dataset and creating dataloader for train and validation phase
    dataset = DetectionDataset(transform=train_transform, grid_size=grid_size, num_bboxes=num_bboxes, path=PATH_TO_DATA)

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
        logger.start_info(optim=optimizer,
                          scheduler=scheduler,
                          hyperparameters={'num_epochs': num_epochs,
                                           'batch_size': batch_size,
                                           'grid_size': grid_size, 'num_bboxes': num_bboxes},
                          transforms=train_transform,
                          comment=COMMENT)

    torch.save(model, os.path.join(PATH_TO_LOG, SESSION_ID, 'model.pt'))
    # Training loop
    for epoch in range(num_epochs):
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

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(image)
                    loss_value, logger_loss = loss(output, target)

                    if phase == 'train':
                        # Parameters updating
                        loss_value.backward()
                        optimizer.step()
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
                'optim_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': batch_train_loss['Total loss']
            }, os.path.join(PATH_TO_LOG, SESSION_ID, 'checkpoint.pt'))
            logger.info('!Checkpoint created!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train detection model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epochs', type=int, default=125, help='number of train epochs')
    parser.add_argument('--batch_size', type=int, default=26, help='size of image batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--grid_size', type=int, default=9, help='grid size')
    parser.add_argument('--num_bboxes', type=int, default=2, help='number of bboxes')
    parser.add_argument('--val_split', type=float, default=0.02, help='validation split')
    parser.add_argument('--path_to_log', type=str, default='log',
                        help='path to log folder.\nlog, model pickle and checkpiont will be accessible in '
                             '"~/<path_to_log>/detection/<session_id>"')
    parser.add_argument('--model', type=str, default='FacedModelLite',
                        choices=['FacedModelLite', 'FacedModel', 'TinyYolo'])
    parser.add_argument('--path_to_data', type=str, default='data/detection/train_images',
                        help='path to folder with data\nfolder with images and markup(txt file) must be accessible in '
                             '"~/<path_to_data>/[train_images|train_markup.txt]"')

    opt = parser.parse_args()
    # Declaring constants for logging and creating dir for current experiment. Initiating logger.
    TASK = 'detection'  # detection or classification
    TEST = False
    PATH_TO_DATA = opt.path_to_data
    PATH_TO_LOG = os.path.join('..', opt.path_to_log, TASK)
    SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
    COMMENT = '{0}, {1}x{1} grid'.format(opt.model, opt.grid_size)

    model_dict = {'FacedModelLite': FacedModelLite, 'FacedModel': FacedModel, 'TinyYolo': TinyYolo}
    net = model_dict[opt.model]
    # Declaring hyperparameters
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    grid_size = opt.grid_size
    model_stride_dict = {'FacedModelLite': 5, 'FacedModel': 6, 'TinyYolo': 6}
    image_size = grid_size * 2 ** model_stride_dict[opt.model]
    image_size = (image_size, image_size)
    num_bboxes = opt.num_bboxes
    val_split = opt.val_split

    run_train()

