import cv2
from utils.transforms import ImageToTensor
import torch
import numpy as np
import os
from utils.utils import xywh2xyxy, from_yolo_target
from data.detection.show_targets import show_rectangles
import time
from torchvision.transforms import ToTensor


PATH_TO_DETECTION_MODEL = 'log\\detection\\20.01.13_12-53'
PATH_TO_RECOGNITION_MODEL = 'log\\emorec'
SIZE = 64
emotions = ['Anger', 'Disgust', 'Neutral', 'Surprise']


detection_model = torch.load(os.path.join(PATH_TO_DETECTION_MODEL, 'model.pt'))
detection_load = torch.load(os.path.join(PATH_TO_DETECTION_MODEL, 'checkpoint.pt'))
detection_model.load_state_dict(detection_load['model_state_dict'])

detection_model.to(torch.device('cpu'))
detection_model.eval()

recognition_model = torch.load(os.path.join(PATH_TO_RECOGNITION_MODEL, 'model.pt'))
recognition_load = torch.load(os.path.join(PATH_TO_RECOGNITION_MODEL, 'checkpoint.pt'))
recognition_model.load_state_dict(recognition_load['model_state_dict'])


recognition_model.eval()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()
    start = time.time()
    image = cv2.resize(image, (320, 320))
    image_t = ImageToTensor()(image)
    image_t = image_t.unsqueeze(0)
    output = detection_model(image_t)
    listed_output = from_yolo_target(output[:, :10, :, :], image_t.size(2), grid_size=5, num_bboxes=2)
    pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]
    pred_xyxy = xywh2xyxy(pred_output[:, :4])
    bbox_h = pred_xyxy[3] - pred_xyxy[1]
    x_center = (pred_xyxy[0] + pred_xyxy[2]) // 2
    face_image = image[pred_xyxy[1]:pred_xyxy[3] + 10, x_center - bbox_h // 2:x_center + bbox_h // 2, :]
    face_image = cv2.resize(face_image, (SIZE, SIZE))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = ToTensor()(face_image).unsqueeze(0)
    pred_emo = recognition_model(face_image).argmax(dim=1).item()
    show_rectangles(image,
                    np.expand_dims(pred_xyxy, axis=0), [emotions[pred_emo]])
    fps = 1. / (time.time() - start)
    print(fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


