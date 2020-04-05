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
from utils.utils import xywh2xyxy, from_yolo_target, bbox_resize
from utils.show_targets import draw_rectangles
import time
from torchvision.transforms import ToTensor


def run_eval():
    # Initialising models weights
    detection_model = torch.load(os.path.join(PATH_TO_DETECTION_MODEL, 'model.pt'))
    detection_load = torch.load(os.path.join(PATH_TO_DETECTION_MODEL, 'checkpoint.pt'))
    detection_model.load_state_dict(detection_load['model_state_dict'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    detection_model.to(device)
    detection_model.eval()

    classification_model = torch.load(os.path.join(PATH_TO_CLASSIFICATION_MODEL, 'model.pt'))
    classification_load = torch.load(os.path.join(PATH_TO_CLASSIFICATION_MODEL, 'checkpoint.pt'))
    classification_model.load_state_dict(classification_load['model_state_dict'])

    classification_model.to(device)
    classification_model.eval()

    cap = cv2.VideoCapture(0)

    # Start video capturing
    while cap.isOpened():
        ret, image = cap.read()  # original image
        orig_shape = image.shape[:2]  # (480, 640)
        start = time.time()
        with torch.no_grad():
            detection_image = cv2.resize(image, DETECTION_SIZE)           # converting image to format and shape,
            detection_image = ImageToTensor()(detection_image).to(device)  # required by detection model
            detection_image = detection_image.unsqueeze(0)
            output = detection_model(detection_image)

            # Convering detection prediction from tensor to list format
            listed_output = from_yolo_target(output[:, :10, :, :], detection_image.size(2), grid_size=GRID_SIZE, num_bboxes=NUM_BBOXES)
            pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]
            pred_xyxy = xywh2xyxy(pred_output[:, :4])  # predicted bbox coordinates

            if pred_output[:, 4][0] > DETECTION_THRESHOLD:  # prediction confidence threshold

                bbox_l_y = int((pred_xyxy[1]) * (orig_shape[0] / DETECTION_SIZE[1]))  # Transform bbox coords
                bbox_r_y = int((pred_xyxy[3]) * (orig_shape[0] / DETECTION_SIZE[1]))  # correspondingly to DETECTION_SHAPE -> orig_shape

                bbox_l_x = int((pred_xyxy[0]) * (orig_shape[1] / DETECTION_SIZE[0]))
                bbox_r_x = int((pred_xyxy[2]) * (orig_shape[1] / DETECTION_SIZE[0]))

                bbox_x_c = (bbox_l_x + bbox_r_x) // 2
                bbox_h = bbox_r_y - bbox_l_y

                bbox_l_x = bbox_x_c - bbox_h // 2  # Make bbox square with sides equal to bbox_h
                bbox_r_x = bbox_x_c + bbox_h // 2

                bbox_l_y = np.clip(bbox_l_y, 0, orig_shape[0])  # clip coordinates which limit image borders
                bbox_r_y = np.clip(bbox_r_y, 0, orig_shape[0])
                bbox_l_x = np.clip(bbox_l_x, 0, orig_shape[1])
                bbox_r_x = np.clip(bbox_r_x, 0, orig_shape[1])

                # Converting image to format and shape required by recognition model
                face_image = image[bbox_l_y:bbox_r_y, bbox_l_x:bbox_r_x, :]

                face_image = cv2.resize(face_image, CLASSIFICATION_SIZE, CLASSIFICATION_SIZE)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = ToTensor()(face_image).unsqueeze(0).to(device)

                pred_emo = classification_model(face_image).argmax(dim=1).item()

                # Paint bbox and emotion prediction
                draw_rectangles(image,
                                np.expand_dims(np.array(bbox_resize(pred_xyxy, DETECTION_SIZE, orig_shape[::-1])), axis=0),
                                [EMOTIONS_LIST[pred_emo]])
            else:
                cv2.imshow('image', image)
            fps = 1. / (time.time() - start)  # Count fps
            print(fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to launc facial emotion recognition pipeline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_detection_model', type=str, default='log/detection/20.03.31_11-56',
                        help='path to detection model')
    parser.add_argument('--path_to_classification_model', type=str, default='log/classification/20.03.30_20-13',
                        help='path to classification model')
    parser.add_argument('--grid_size', type=int, default=9, help='grid size')
    parser.add_argument('--num_bboxes', type=int, default=2, help='number of bboxes')
    parser.add_argument('--detection_shape', type=int, default=288, help='detection image size')
    parser.add_argument('--classification_shape', type=int, default=64, help='classification image size')
    parser.add_argument('--detection_threshold', type=float, default=0.4, help='detection threshold')
    parser.add_argument('--emotions', type=str, default='Anger Happy Neutral Surprise',
                        help='emotions list which model has trained on (space separated)\n')
    opt = parser.parse_args()

    # Declaring paths to models and hyperparameters
    PATH_TO_DETECTION_MODEL = os.path.join('..', opt.path_to_detection_model)
    PATH_TO_CLASSIFICATION_MODEL = os.path.join('..', opt.path_to_recognition_model)
    GRID_SIZE = opt.grid_size
    NUM_BBOXES = opt.num_bboxes
    EMOTIONS_LIST = opt.emotions.split()
    DETECTION_SIZE = (opt.detection_shape, opt.detection_shape)
    CLASSIFICATION_SIZE = (opt.recognition_shape, opt.recognition_shape)
    DETECTION_THRESHOLD = opt.detection_threshold

    run_eval()
