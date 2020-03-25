import albumentations
import torch
from torch.utils.data import DataLoader
from utils.loss import Loss
from utils.datasets import DetectionDataset
import os


def get_size_model(model):
    # return the size of model in MB
    torch.save(model.state_dict(), "temp.pt")
    size = os.path.getsize("temp.pt") / 1e6
    os.remove("temp.pt")
    return size

# POST TRAINING QUANTIZATION

PATH_TO_MODEL_FLOAT = 'log/detection/20.03.25_18-50'
image_size = (320, 320)
batch_size = 1
num_calibration_batches = 5


model = torch.load(os.path.join(PATH_TO_MODEL_FLOAT, 'model.pt')).to('cpu')
model.eval()

print("Size of model before quntization:", get_size_model(model))

model.fuse_model()
# model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)

loss = Loss(model.S, model.B)


transforms = albumentations.Compose([
    albumentations.RandomSizedBBoxSafeCrop(height=image_size[0], width=image_size[1], always_apply=True),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Rotate(15, p=0.5),

], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=transforms, grid_size=model.S, num_bboxes=model.B)
dataloader = DataLoader(dataset, batch_size=batch_size)

for i, (image, target, _) in enumerate(dataloader):
    if i > num_calibration_batches:
        break
    with torch.no_grad():
        output = model(image)
        loss(output, target)


torch.quantization.convert(model, inplace=True)

print("Size of model after quntization:", get_size_model(model))
