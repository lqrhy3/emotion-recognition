import os
import albumentations
import torch
from torch.utils.data import DataLoader
from utils.datasets import EmoRecDataset
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from utils.transforms import GaussNoise


# Quantization aware training
PATH_TO_DATA = 'data/classification/callibration_images'
PATH_TO_MODEL = 'log/emorec/20.03.30_20-13/model.pt'
PATH_TO_STATE_DICT = 'log/emorec/20.03.30_20-13/checkpoint.pt'

ENGINE = 'fbgemm'
IMAGE_SIZE = (64, 64)
NUM_BATCHES = 10
BATCH_SIZE = 6
NUM_EPOCHS = 10
SAVE_TYPE = 'trace'
EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Surprise']

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    GaussNoise(var_limit=(0, 100), p=0.7),
    transforms.Grayscale(),
    transforms.ToTensor()
])

dataset = EmoRecDataset(path=PATH_TO_DATA, emotions=EMOTIONS, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = torch.load(os.path.join('..', PATH_TO_MODEL), map_location='cpu')
load = torch.load(os.path.join('..', PATH_TO_STATE_DICT), map_location='cpu')
model.load_state_dict(load['model_state_dict'])


model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
torch.quantization.prepare_qat(model, inplace=True)

model.train()

loss = CrossEntropyLoss(reduction='mean')

# ???????????????????????????????????????
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=1)  # OFF


for epoch in range(NUM_EPOCHS):
    print("Epochs:", epoch)
    for image, label in dataloader:

        optimizer.zero_grad()

        output = model(image)

        loss_value = loss(output, label)
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
    torch.jit.save(torch.jit.trace(model, torch.ones((BATCH_SIZE, 1, *IMAGE_SIZE))),
                   os.path.join('..', PATH_TO_MODEL[:-8], PATH_TO_MODEL.split('/')[-1][:-3] + '_quantized.pt'))
elif SAVE_TYPE == 'script':
    torch.jit.save(torch.jit.script(model),
                   os.path.join('..', PATH_TO_MODEL[:-8], PATH_TO_MODEL.split('/')[-1][:-3] + '_quantized.pt'))
