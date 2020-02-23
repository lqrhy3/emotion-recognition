from .base_camera import BaseCamera
from .pipeline import stream_prediction
import cv2
import torch
import os
import rootpath


PATH_TO_DETECTION_MODEL = 'log\\detection\\20.01.13_12-53'
PATH_TO_RECOGNITION_MODEL = 'log\\emorec\\20.01.25_03-16'
PATH_TO_DETECTION_MODEL = os.path.join(rootpath.detect(), PATH_TO_DETECTION_MODEL)
PATH_TO_RECOGNITION_MODEL = os.path.join(rootpath.detect(), PATH_TO_RECOGNITION_MODEL)
emotions = ['Anger', 'Happy', 'Neutral', 'Surprise']
DETECTION_SHAPE = (320, 320)
EMOREC_SHAPE = (64, 64)
DETECTION_TRESHOLD = 0.4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

detection_model = torch.load(os.path.join(PATH_TO_DETECTION_MODEL, 'model.pt'), map_location=device)
detection_load = torch.load(os.path.join(PATH_TO_DETECTION_MODEL, 'checkpoint.pt'), map_location=device)
detection_model.load_state_dict(detection_load['model_state_dict'])


detection_model.to(device)
detection_model.eval()

recognition_model = torch.load(os.path.join(PATH_TO_RECOGNITION_MODEL, 'detection_model.pt'), device)
recognition_load = torch.load(os.path.join(PATH_TO_RECOGNITION_MODEL, 'checkpoint.pt'), device)
recognition_model.load_state_dict(recognition_load['model_state_dict'])

recognition_model.to(device)
recognition_model.eval()


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, image = camera.read()
            yield stream_prediction(image, detection_model, recognition_model, device)

