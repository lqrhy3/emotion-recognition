import os
import albumentations
import torch
from torch.utils.data import DataLoader
from utils.datasets import DetectionDataset
from utils.loss import Loss


# Quantization aware training
PATH_TO_DATA = 'data/detection/callibration_images'
PATH_TO_MODEL = 'log/detection/20.03.26_17-52/model.pt'
PATH_TO_STATE_DICT = 'log/detection/20.03.26_17-52/checkpoint.pt'

ENGINE = 'fbgemm'
IMAGE_SIZE = (288, 288)
NUM_BATCHES = 10
BATCH_SIZE = 26
NUM_EPOCHS = 10
GRID_SIZE = 9
SAVE_TYPE = 'trace'

train_transform = albumentations.Compose([
    albumentations.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], always_apply=True),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Rotate(15, p=0.5),

], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = DetectionDataset(transform=train_transform, grid_size=9, num_bboxes=2)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = torch.load(os.path.join('..', PATH_TO_MODEL), map_location='cpu')
load = torch.load(os.path.join('..', PATH_TO_STATE_DICT), map_location='cpu')
model.load_state_dict(load['model_state_dict'])
model.load_state_dict(load)

model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
torch.quantization.prepare_qat(model, inplace=True)

model.train()

loss = Loss(grid_size=GRID_SIZE, num_bboxes=2)

# ???????????????????????????????????????
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=1)  # OFF


for epoch in range(NUM_EPOCHS):
    for image, target, _ in dataloader:
        optimizer.zero_grad()

        output = model(image)

        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        scheduler.step(epoch)

    if epoch > 2:
        # Freeze batch norm mean and variance estimates
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    if epoch > 3:
        # Freeze quantizer parameters
        model.apply(torch.quantization.disable_observer)

model.eval()
torch.quantization.convert(model, inplace=True)

if SAVE_TYPE == 'trace':
    torch.jit.save(torch.jit.trace(model),
                   torch.ones((BATCH_SIZE, 3, *IMAGE_SIZE)),
                   os.path.join(PATH_TO_MODEL, PATH_TO_MODEL.split('/')[-1][:-3] + 'quantized.pt'))
elif SAVE_TYPE == 'script':
    torch.jit.save(torch.jit.script(model),
                   os.path.join(PATH_TO_MODEL, PATH_TO_MODEL.split('/')[-1][:-3] + 'quantized.pt'))
