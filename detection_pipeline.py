import cv2
from utils.transforms import ImageToTensor
import torch
from models.tiny_yolo_model import TinyYolo
from models.faced_model import FacedModel
import numpy as np
import os
from utils.utils import xywh2xyxy, from_yolo_target
from data.detection.show_targets import show_rectangles
import time


# Initialising detection model
PATH_TO_MODEL = 'log/detection/20.03.26_17-52'

# model = TinyYolo(grid_size=5, num_bboxes=2, n_classes=1)
# model = FacedModel(grid_size=5, num_bboxes=2, n_classes=1)

# model = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
# load = torch.load(os.path.join(PATH_TO_MODEL, 'checkpoint.pt'), map_location='cpu')
# model.load_state_dict(load['model_state_dict'])
# model.to('cpu')
model = torch.jit.load(os.path.join(PATH_TO_MODEL, 'model_quantized.zip'))
model.eval()

cap = cv2.VideoCapture(0)

while cap.isOpened():  # Capturing video
    ret, image = cap.read()
    start = time.time()

    # Image preprocessing for format and shape required by model
    image = cv2.resize(image, (320, 320))
    image = ImageToTensor()(image)
    image = image.unsqueeze(0)
    output = model(image)  # Prediction

    listed_output = from_yolo_target(output[:, :10, :, :], image.size(2), grid_size=5, num_bboxes=2)  # Converting from tensor format to list
    pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]  # Selecting most confident cell
    show_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)),
                    np.expand_dims(xywh2xyxy(pred_output[:, :4]), axis=0), str(pred_output[:, 4]))  # Painting bbox
    fps = 1. / (time.time() - start)
    print(fps)
    print(ret)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


