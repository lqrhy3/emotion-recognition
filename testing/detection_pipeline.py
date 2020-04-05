import argparse
import cv2
import os

import sys
module_path = os.path.abspath(os.getcwd() + '\\..')
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.transforms import ImageToTensor
import torch
import numpy as np
from utils.utils import xywh2xyxy, from_yolo_target
from utils.show_targets import draw_rectangles
import time


def run_eval():
    model = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
    load = torch.load(os.path.join(PATH_TO_MODEL, 'checkpoint.pt'), map_location='cpu')
    model.load_state_dict(load['model_state_dict'])
    # model = torch.jit.load(os.path.join(PATH_TO_MODEL, 'model_quantized.pt'))
    model.to('cpu').eval()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():  # Capturing video
        ret, image = cap.read()
        start = time.time()

        # Image preprocessing for format and shape required by model
        image = cv2.resize(image, image_size)
        image = ImageToTensor()(image)
        image = image.unsqueeze(0)
        output = model(image)  # Prediction

        listed_output = from_yolo_target(output[:, :10, :, :], image.size(2), grid_size=grid_size, num_bboxes=num_bboxes)  # Converting from tensor format to list
        pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]  # Selecting most confident cell
        draw_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)),
                        np.expand_dims(xywh2xyxy(pred_output[:, :4]), axis=0),
                        str(pred_output[:, 4]))  # Painting bbox
        fps = 1. / (time.time() - start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to evaluate detection model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_model', type=str, default='log/detection/20.03.31_11-56', help='path to model')
    parser.add_argument('--grid_size', type=int, default=9, help='grid size')
    parser.add_argument('--num_bboxes', type=int, default=2, help='number of bboxes')
    parser.add_argument('--image_size', type=int, default=288, help='image size')
    opt = parser.parse_args()
    # Initialising detection model
    PATH_TO_MODEL = os.path.join('..', opt.path_to_model)
    image_size = (opt.image_size, opt.image_size)
    grid_size = opt.grid_size
    num_bboxes = opt.num_bboxes

    run_eval()
