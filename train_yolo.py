import torch
from yolo_model import TinyYolo
from torchvision import transforms
from utils.logger import create_logger

"""Initiate logger"""
logger = create_logger('Yolo')


"""Data config"""
train_path = ''
test_path = ''

"""Initiate model"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TinyYolo().to(device)

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])




