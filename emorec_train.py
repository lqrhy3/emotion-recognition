from emorec_model import *
import numpy as np
import datetime
import os
from utils.logger import Logger
from utils.datasets import EmoRecDataset
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import CrossEntropyLoss
import torch
import warnings

warnings.filterwarnings('ignore')
# Declaring constants for logging and creating dir for current experiment. Initiating logger.
TASK = 'emorec'
TEST = False
PATH_TO_LOG = 'log/' + TASK
SESSION_ID = datetime.datetime.now().strftime('%y.%m.%d_%H-%M')
COMMENT = 'MiniXception FER'

# Declaring hyperparameters
n_epoch = 20
batch_size = 16
val_split = 0.05
lr = 0.001
emotions = ['Anger', 'Disgust', 'Neutral', 'Surprise']

# Initiating model and device (cuda/cpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MiniXception(emotion_map=emotions).to(device)

optim = torch.optim.Adam(model.parameters(), lr=lr)

train_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

dataset = EmoRecDataset(transform=train_transforms, path='../data/fer')

dataset_len = len(dataset)
val_len = int(np.floor(val_split * dataset_len))
val_idxs = np.random.choice(list(range(dataset_len)), val_len)
train_idxs = list(set(list(range(dataset_len))) - set(val_idxs))

train_sampler = SubsetRandomSampler(train_idxs)
val_sampler = SubsetRandomSampler(val_idxs)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)

loss = CrossEntropyLoss(reduction='mean')


if not TEST:
    os.makedirs(os.path.join(PATH_TO_LOG, SESSION_ID), exist_ok=True)
    logger = Logger('ConvNet for EmoRec', task=TASK, session_id=SESSION_ID)

    hyperparameters = {'n_epoch': n_epoch, 'batch_size': batch_size, 'emotions': emotions}
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

        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)

            optim.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                pred = model(image)
                pred_class = pred.argmax(dim=1)
                loss_value = loss(pred, label)

                if phase == 'train':
                    loss_value.backward()
                    optim.step()
                    batch_train_loss += loss_value.item()
                else:
                    batch_val_loss += loss_value.item()
                    batch_val_metric += (pred_class == label.data).float().mean().item()

    epoch_train_loss = batch_train_loss / len(train_dataloader)
    epoch_val_loss = batch_val_loss / len(val_dataloader)
    epoch_val_metric = batch_val_metric / len(val_dataloader)

    if not TEST:
        # Saving
        logger.epoch_info(epoch=epoch, train_loss=epoch_train_loss,
                          val_loss=epoch_val_loss, val_metrics=epoch_val_metric)
        if epoch % 5 == 0:
            # Checkpoint. Train info
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'loss': epoch_train_loss
            }, os.path.join(PATH_TO_LOG, SESSION_ID, 'checkpoint.pt'))
            logger.info('!Checkpoint created!')
