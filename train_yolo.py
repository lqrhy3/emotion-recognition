import torch
from yolo_model import TinyYolo
from torchvision import transforms
from utils.logger import Logger
from utils.datasets import DetectionDataset
"""Initiate logger"""
logger = Logger('Yolo')


"""Initiate model"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TinyYolo().to(device)

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

a = DetectionDataset()
for i in range(len(a)):
    sample = a[i]
    print(i, sample[1])



