import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils.datasets import EmoRecDataset
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from utils.transforms import GaussNoise


# Quantization aware training
def run_qat():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        GaussNoise(var_limit=(0, 100), p=0.7),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = EmoRecDataset(path=PATH_TO_DATA, emotions=EMOTIONS_LIST, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
    load = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
    model.load_state_dict(load['model_state_dict'])

    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
    torch.quantization.prepare_qat(model, inplace=True)

    model.train()

    loss = CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=1)  # OFF

    for epoch in range(NUM_EPOCHS):
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

    load['model_state_dict'] = model.state_dict()
    torch.save(load, os.path.join(PATH_TO_MODEL, 'checkpoint_qat.pt'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to run quantization aware training '
                                                 'for classification task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_model', type=str, default='log/emorec/20.03.30_20-13', help='path to model')
    parser.add_argument('--path_to_data', type=str, default='data/classification/callibration_images',
                        help='path to data')
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=12, help='calibration batch size')
    parser.add_argument('--image_size', type=int, default=64, help='calibration image size')
    parser.add_argument('--emotions', type=str, default='Anger Happy Neutral Surprise',
                        help='emotions list which model has trained on (space separated)')
    opt = parser.parse_args()

    PATH_TO_MODEL = os.path.join('..', opt.path_to_model)
    PATH_TO_DATA = os.path.join('..', opt.path_to_data)
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    IMAGE_SIZE = opt.image_size
    NUM_BATCHES = opt.num_batches
    EMOTIONS_LIST = opt.emotions
    ENGINE = 'fbgemm'

    run_qat()
