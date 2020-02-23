import cv2
import numpy as np
from utils.utils import xywh2xyxy, from_yolo_target
from torchvision.transforms import ToTensor
from utils.transforms import ImageToTensor
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

DETECTION_SHAPE = config['Constants']['detection_shape']
EMOREC_SHAPE = config['Constants']['emorec_shape']
DETECTION_TRESHOLD = config['Constants']['detection_threshold']

emotions = config['Emotions']['emotions']


def bbox_resize(coords, from_shape, to_shape):
    new_coords = list()
    new_coords.append(int(coords[0] * (to_shape[0] / from_shape[0])))
    new_coords.append(int(coords[1] * (to_shape[1] / from_shape[1])))
    new_coords.append(int(coords[2] * (to_shape[0] / from_shape[0])))
    new_coords.append(int(coords[3] * (to_shape[1] / from_shape[1])))
    return new_coords


def stream_prediction(image, detection_model, recognition_model, device):
    orig_shape = image.shape[:2]  # (480, 640)

    detection_image = cv2.resize(image, DETECTION_SHAPE)
    detection_image = ImageToTensor()(detection_image).to(device)
    detection_image = detection_image.unsqueeze(0)
    output = detection_model(detection_image)

    listed_output = from_yolo_target(output[:, :10, :, :], detection_image.size(2), grid_size=5, num_bboxes=2)
    pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]
    pred_xyxy = xywh2xyxy(pred_output[:, :4])

    if pred_output[:, 4][0] > DETECTION_TRESHOLD:

        bbox_h = pred_xyxy[3] - pred_xyxy[1]
        bbox_x_center = (pred_xyxy[0] + pred_xyxy[2]) // 2

        bbox_l_y = int(
            (pred_xyxy[1]) * (orig_shape[0] / DETECTION_SHAPE[1]))  # Make bbox square with sides equal to
        bbox_r_y = int(
            (pred_xyxy[3]) * (orig_shape[0] / DETECTION_SHAPE[1]))  # bbox height. And transform its coords

        bbox_l_x = int(
            (bbox_x_center - bbox_h // 2) * (orig_shape[1] / DETECTION_SHAPE[0]))  # correspondingly to
        bbox_r_x = int((bbox_x_center + bbox_h // 2) * (
                orig_shape[1] / DETECTION_SHAPE[0]))  # DETECTION_SHAPE -> orig_shape

        bbox_l_y = np.clip(bbox_l_y, 0, orig_shape[0])
        bbox_r_y = np.clip(bbox_r_y, 0, orig_shape[0])
        bbox_l_x = np.clip(bbox_l_x, 0, orig_shape[1])
        bbox_r_x = np.clip(bbox_r_x, 0, orig_shape[1])

        face_image = image[bbox_l_y:bbox_r_y, bbox_l_x:bbox_r_x, :]
        face_image = cv2.resize(face_image, EMOREC_SHAPE, EMOREC_SHAPE)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = ToTensor()(face_image).unsqueeze(0).to(device)

        pred_emo = recognition_model(face_image).argmax(dim=1).item()

        """"""
        rectangles = np.expand_dims(np.array(bbox_resize(pred_xyxy, DETECTION_SHAPE, orig_shape[::-1])), axis=0)
        emotions_list = [emotions[pred_emo]]
        color = (0, 255, 0)
        thickness = 2

        if not emotions_list:
            emotions_list = [''] * rectangles.shape[0]
        for rectangle, emotion in zip(rectangles, emotions_list):
            image = cv2.rectangle(image, (int((rectangle[0] + rectangle[2]) / 2 - (rectangle[3] - rectangle[1]) / 2),
                                          rectangle[1]), (
                                  int((rectangle[0] + rectangle[2]) / 2 + (rectangle[3] - rectangle[1]) / 2),
                                  rectangle[3]), color, thickness)
            image = cv2.putText(image, emotion, (rectangle[0], rectangle[1] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), thickness)
        return cv2.imencode('.jpg', image)[1].tobytes()

    else:
        return cv2.imencode('.jpg', image)[1].tobytes()
