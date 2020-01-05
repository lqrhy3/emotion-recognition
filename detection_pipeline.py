import cv2
from utils.transforms import ImageToTensor
import torch
from tiny_yolo_model import TinyYolo
import numpy as np
import os
from utils.utils import xywh2xyxy, from_yolo_target
from data.detection.show_targets import show_rectangles
import time


PATH_TO_MODEL = 'checkpoint.pt'

model = TinyYolo(grid_size=7, num_bboxes=2, n_classes=1)
load = torch.load(os.path.join('log\\detection', '20.01.05_13-59', PATH_TO_MODEL))
model.load_state_dict(load['model_state_dict'])


model.eval()


cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, image = cap.read()
    start = time.time()
    image = cv2.resize(image, (448, 448))
    image = ImageToTensor()(image)
    image = image.unsqueeze(0)
    output = model(image)
    listed_output = from_yolo_target(output[:, :10, :, :], image.size(2), grid_size=7, num_bboxes=2)
    pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]
    show_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)),
                    np.expand_dims(xywh2xyxy(pred_output[:, :4]), axis=0), pred_output[:, 4])
    fps = 1. / (time.time() - start)
    print(fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


