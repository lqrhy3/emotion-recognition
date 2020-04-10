import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


# POST TRAINING QUANTIZATION
def run_ptq():
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    dataset = ImageFolder(os.path.join(PATH_TO_DATA), transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
    load = torch.load(os.path.join(PATH_TO_MODEL, 'checkpoint.pt'), map_location='cpu')
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
                       os.path.join(PATH_TO_MODEL, 'model_quantized.pt'))
    elif SAVE_TYPE == 'script':
        torch.jit.save(torch.jit.script(model),
                       os.path.join(PATH_TO_MODEL, 'model_quantized.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run post training quantization with calibration '
                                                 'for classification task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_model', type=str, default='log/emorec/20.03.30_20-13', help='path to model')
    parser.add_argument('--path_to_data', type=str, default='data/classification/callibration_images',
                        help='path to calibration data')
    parser.add_argument('--image_size', type=int, default=64, help='calibration image size')
    parser.add_argument('--batch_size', type=int, default=1, help='calibration batch size')
    parser.add_argument('--num_batches', type=int, default=5, help='number of calibration batches')
    parser.add_argument('--emotions', type=str, default='Anger Happy Neutral Surprise',
                        help='emotions list which model has trained on (space separated)')
    parser.add_argument('--save_type', type=str, default='trace',
                        help='whether to use JIT.trace or to JIT.script to save quantized model',
                        choices=['trace', 'script'])
    opt = parser.parse_args()

    PATH_TO_MODEL = os.path.join('..', opt.path_to_model)
    PATH_TO_DATA = os.path.join('..', opt.path_to_data)
    IMAGE_SIZE = opt.image_size
    BATCH_SIZE = opt.batch_size
    NUM_BATCHES = opt.num_batches
    EMOTIONS_LIST = opt.emotions.split()
    SAVE_TYPE = opt.save_type
    ENGINE = 'fbgemm'

    run_ptq()
