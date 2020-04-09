import argparse
import os
import albumentations
import torch
from torch.utils.data import DataLoader
from utils.datasets import DetectionDataset
from utils.loss import Loss


# Quantization aware training
def run_qat():
    train_transform = albumentations.Compose([
        albumentations.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], always_apply=True),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(15, p=0.5),

    ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))

    dataset = DetectionDataset(transform=train_transform, grid_size=9, num_bboxes=2, path=PATH_TO_DATA)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
    load = torch.load(os.path.join(PATH_TO_MODEL, 'checkpoint.pt'), map_location='cpu')
    model.load_state_dict(load['model_state_dict'])

    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig(ENGINE)
    torch.quantization.prepare_qat(model, inplace=True)

    model.train()

    loss = Loss(grid_size=GRID_SIZE, num_bboxes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=1)  # OFF

    for epoch in range(NUM_EPOCHS):
        print("Epochs:", epoch)
        for image, target, _ in dataloader:
            optimizer.zero_grad()

            output = model(image)

            loss_value, _ = loss(output, target)
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
    parser = argparse.ArgumentParser(description='Script to quantization aware training'
                                                 'for detection task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_model', type=str, default='log/detection/20.03.26_12-28', help='path to model')
    parser.add_argument('--path_to_data', type=str, default='data/detection/callibration_images')
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=26, help='calibration batch size')
    parser.add_argument('--grid_size', type=int, default=9)
    parser.add_argument('--num_bboxes', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=288, help='calibration image size')
    opt = parser.parse_args()

    PATH_TO_MODEL = os.path.join('..', opt.path_to_model)
    PATH_TO_DATA = os.path.join('..', opt.path_to_data)
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    GRID_SIZE = opt.grid_size
    NUM_BBOXES = opt.num_bboxes
    IMAGE_SIZE = opt.image_size
    NUM_BATCHES = opt.num_batches
    ENGINE = 'fbgemm'

    run_qat()
