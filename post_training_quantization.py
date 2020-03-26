import albumentations
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.loss import Loss
from utils.datasets import EmoRecDataset, DetectionDataset
from torch.nn import CrossEntropyLoss
import os


def calibration_detection(model, dataloader, loss, num_calibration_batches=5):
    for i, (image, target, _) in enumerate(dataloader):
        if i > num_calibration_batches:
            break
        with torch.no_grad():
            output = model(image)
            loss(output, target)


def calibration_recognition(model, dataloader, loss, num_calibration_batches=5):
    for i, (image, target) in enumerate(dataloader):
        if i > num_calibration_batches:
            break
        with torch.no_grad():
            output = model(image)
            loss(output, target)


# POST TRAINING QUANTIZATION

# MODEL_TO_QUANTIZE = 'recognition'
MODEL_TO_QUANTIZE = 'detection'
PATH_TO_MODEL_FLOAT = 'log/detection/20.03.26_17-52'

image_size = (320, 320)
batch_size = 1
num_calibration_batches = 5

model = torch.load(os.path.join(PATH_TO_MODEL_FLOAT, 'model.pt'), map_location='cpu')
load = torch.load(os.path.join(PATH_TO_MODEL_FLOAT, 'checkpoint.pt'), map_location='cpu')
model.load_state_dict(load['model_state_dict'])
model.eval()

model.fuse_model()
# model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)

emotions = ['Anger', 'Happy', 'Neutral', 'Surprise']

transforms_detection = albumentations.Compose([
    albumentations.RandomSizedBBoxSafeCrop(height=image_size[0], width=image_size[1], always_apply=True),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Rotate(15, p=0.5),
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

transforms_recognition = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.ToTensor()
])

dataset_recognition = EmoRecDataset(transform=transforms_recognition, path='data/fer', emotions=emotions)
dataset_detection = DetectionDataset(transform=transforms_detection, grid_size=model.S, num_bboxes=model.B)

if MODEL_TO_QUANTIZE == 'detection':
    dataloader = DataLoader(dataset_detection, batch_size=batch_size)
    calibration_detection(model, dataloader, Loss(grid_size=model.S, num_bboxes=model.B))
else:
    dataloader = DataLoader(dataset_recognition, batch_size=batch_size)
    calibration_recognition(model, dataloader, CrossEntropyLoss(reduction='mean'))

torch.quantization.convert(model, inplace=True)

# torch.save(model.state_dict(), os.path.join(PATH_TO_MODEL_FLOAT, 'state_dict_quantized.pt'))

# torch.jit.save(torch.jit.trace(model, torch.ones((1, 3, 320, 320))),
#                os.path.join(PATH_TO_MODEL_FLOAT, 'model_trace_quantized.pt'))


torch.jit.save(torch.jit.script(model), os.path.join(PATH_TO_MODEL_FLOAT, 'model_quantized.zip'))
