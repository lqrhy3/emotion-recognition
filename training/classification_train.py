import argparse
import datetime
import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import sys
module_path = os.path.abspath(os.getcwd() + '\\..')
if module_path not in sys.path:
    sys.path.append(module_path)

from models.classification.mini_xception import *
from models.classification.conv_net import ConvNet
from utils.logger import Logger
from utils.datasets import EmoRecDataset
from utils.transforms import GaussNoise
import warnings

warnings.filterwarnings('ignore')


def run_train():
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        GaussNoise(var_limit=(0, 100), p=0.7),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Initialising dataset and dataloaders for train/validation stages
    dataset = EmoRecDataset(transform=train_transforms, path=PATH_TO_DATA, emotions=emotion_list)
    emotions = dataset.classes
    dataset_len = len(dataset)
    val_len = int(np.floor(val_split * dataset_len))
    val_idxs = np.random.choice(list(range(dataset_len)), val_len)
    train_idxs = list(set(list(range(dataset_len))) - set(val_idxs))

    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)

    loss = CrossEntropyLoss(reduction='mean')

    # Initiating model and device (cuda/cpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = net(emotion_map=emotions, in_channels=dataset.img_channels).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if not TEST:
        os.makedirs(os.path.join(PATH_TO_LOG, SESSION_ID), exist_ok=True)
        logger = Logger(COMMENT, task=TASK, session_id=SESSION_ID)

        hyperparameters = {'num_epochs': n_epoch, 'batch_size': batch_size, 'emotions': emotions}
        logger.start_info(hyperparameters=hyperparameters, optim=optim, transforms=train_transforms, comment=COMMENT)

    torch.save(model, os.path.join(PATH_TO_LOG, SESSION_ID, 'model.pt'))
    # Training loop
    for epoch in range(n_epoch):
        batch_train_loss = 0
        batch_val_loss = 0
        batch_val_metric = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = val_dataloader
                model.eval()

            for i, (image, label) in enumerate(dataloader):
                # cv2.imshow('name', image[0].detach().numpy().transpose(1, 2, 0))
                # cv2.waitKey(0)
                image = image.to(device)
                label = label.to(device)

                optim.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(image)
                    pred_class = pred.argmax(dim=1)
                    loss_value = loss(pred, label)

                    if phase == 'train':  # Train step
                        loss_value.backward()
                        optim.step()
                        batch_train_loss += loss_value.item()
                    else:  # Validation metrics computing
                        batch_val_loss += loss_value.item()
                        batch_val_metric += (pred_class == label.data).float().mean().item()

        epoch_train_loss = batch_train_loss / len(train_dataloader)
        epoch_val_loss = batch_val_loss / len(val_dataloader)
        epoch_val_metric = batch_val_metric / len(val_dataloader)

        if not TEST:
            # Logger update
            logger.epoch_info(epoch=epoch, train_loss=epoch_train_loss,
                              val_loss=epoch_val_loss, val_metrics=epoch_val_metric)
            if epoch % 5 == 0:
                # Save checkpoint. Model weights and train info
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'loss': epoch_train_loss
                }, os.path.join(PATH_TO_LOG, SESSION_ID, 'checkpoint.pt'))
                logger.info('!Checkpoint created!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train classification model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epochs', type=int, default=211, help='number of train epochs')
    parser.add_argument('--batch_size', type=int, default=63, help='size of image batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--val_split', type=float, default=0.05, help='validation split')
    parser.add_argument('--path_to_log', type=str, default='log',
                        help='path to log folder.\nlog, model pickle and checkpiont will be accessible in '
                             '"~/<path_to_log>/classification/<session_id>"')
    parser.add_argument('--comment', type=str, default=None, help='comment for logger')
    parser.add_argument('--model', type=str, default='MiniXception',
                        choices=['MiniXception', 'ConvNet'])
    parser.add_argument('--path_to_data', type=str, default='data/classification/train_images',
                        help='path to folder with data\nfolder with images must be accessible in "~/<path_to_data>/"\n'
                             'folder <path to data> must contain folders named regard images class')
    parser.add_argument('--emotions', type=str, default='Anger Happy Neutral Surprise',
                        help='emotions to classify\nshould be name of folders in folder <path_to_data>'
                             ' (space separated)\nif None - emotions to classify - all classes in datafolder')

    opt = parser.parse_args()

    # Declaring constants for logging and creating dir for current experiment. Initiating logger.
    TASK = 'classification'
    TEST = False
    PATH_TO_LOG = os.path.join('..', opt.path_to_log, TASK)
    PATH_TO_DATA = opt.path_to_data
    SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
    COMMENT = opt.comment

    # Declaring hyperparameters
    n_epoch = opt.num_epochs
    batch_size = opt.batch_size
    val_split = opt.val_split
    lr = opt.lr

    model_dict = {'ConvNet': ConvNet, 'MiniXception': MiniXception}
    net = model_dict[opt.model]
    emotion_list = opt.emotions.split()
    run_train()
