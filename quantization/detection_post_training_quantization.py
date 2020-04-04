import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils.datasets import DetectionDataset
import albumentations

# POST TRAINING QUANTIZATION
PATH_TO_DATA = 'data/detection/callibration_images'
PATH_TO_MODEL = 'log/detection/20.03.26_12-28/model.pt'
PATH_TO_STATE_DICT = 'log/detection/20.03.26_12-28/checkpoint.pt'

IMAGE_SIZE = (288, 288)
BATCH_SIZE = 1
NUM_BATCHES = 5
ENGINE = 'fbgemm'
SAVE_TYPE = 'script'

transform = albumentations.Compose([
    albumentations.Resize(*IMAGE_SIZE),
], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))


dataset = DetectionDataset(path=os.path.join(PATH_TO_DATA), transform=transform, num_bboxes=2, grid_size=9)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

model = torch.load(os.path.join('..', PATH_TO_MODEL), map_location='cpu')
load = torch.load(os.path.join('..', PATH_TO_STATE_DICT), map_location='cpu')
model.load_state_dict(load['model_state_dict'])

model.eval()

model.fuse_model()
torch.backends.quantized.engine = ENGINE
model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
torch.quantization.prepare(model, inplace=True)

with torch.no_grad():
    for i, (image, label, _) in enumerate(dataloader, 1):
        image = torch.tensor(image, dtype=torch.float)
        image.requires_grad = False
        model(image)
        if i >= NUM_BATCHES:
            break

torch.quantization.convert(model, inplace=True)

if SAVE_TYPE == 'trace':
    torch.jit.save(torch.jit.trace(model, torch.ones((BATCH_SIZE, 3, *IMAGE_SIZE))),
                   os.path.join('..', PATH_TO_MODEL[:-8], PATH_TO_MODEL.split('/')[-1][:-3] + '_quantized.pt'))
elif SAVE_TYPE == 'script':
    torch.jit.save(torch.jit.script(model),
                   os.path.join('..', PATH_TO_MODEL[:-8], PATH_TO_MODEL.split('/')[-1][:-3] + '_quantized.pt'))
