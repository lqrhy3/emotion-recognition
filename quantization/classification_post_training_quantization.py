import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# POST TRAINING QUANTIZATION
PATH_TO_DATA = 'data/classification/callibration_images'
PATH_TO_MODEL = 'log/emorec/20.03.30_20-13/model.pt'
PATH_TO_STATE_DICT = 'log/emorec/20.03.30_20-13/checkpoint.pt'

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 1
NUM_BATCHES = 5
ENGINE = 'fbgemm'
SAVE_TYPE = 'trace'
EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Surprise']

transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.Grayscale(),
                                transforms.ToTensor()])

dataset = ImageFolder(os.path.join('..', PATH_TO_DATA), transform=transform)
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
    for i, (image, label) in enumerate(dataloader, 1):
        image = torch.tensor(image, dtype=torch.float)
        image.requires_grad = False
        model(image)
        if i >= NUM_BATCHES:
            break

torch.quantization.convert(model, inplace=True)

if SAVE_TYPE == 'trace':
    torch.jit.save(torch.jit.trace(model, torch.ones((BATCH_SIZE, 1, *IMAGE_SIZE))),
                   os.path.join('..', PATH_TO_MODEL[:-8], PATH_TO_MODEL.split('/')[-1][:-3] + '_quantized.pt'))
elif SAVE_TYPE == 'script':
    torch.jit.save(torch.jit.script(model),
                   os.path.join('..', PATH_TO_MODEL[:-8], PATH_TO_MODEL.split('/')[-1][:-3] + '_quantized.pt'))


