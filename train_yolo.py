import torch
from yolo_model import TinyYolo
from torchvision import transforms
from utils.logger import create_logger
from utils.datasets import DetectDataset
"""Initiate logger"""
logger = create_logger('Yolo')



"""Initiate model"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TinyYolo().to(device)

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

a = DetectDataset()
for i in range(len(a)):
    sample = a[i]
    print(sample[0], sample[1])



