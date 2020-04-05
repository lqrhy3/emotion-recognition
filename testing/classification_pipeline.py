import argparse
import cv2
from torchvision.transforms import ToTensor
import torch
import numpy as np
import os
import time


def get_subframe_coords(frame_w, frame_h, subframe_w, subframe_h):
    x_l = frame_w // 2 - subframe_w // 2
    y_t = frame_h // 2 - subframe_h // 2
    x_r = frame_w // 2 + subframe_w // 2
    y_b = frame_h // 2 + subframe_h // 2

    return x_l, y_t, x_r, y_b


def run_eval():
    model = torch.load(os.path.join(PATH_TO_MODEL, 'model.pt'), map_location='cpu')
    load = torch.load(os.path.join(PATH_TO_MODEL, 'checkpoint.pt'), map_location='cpu')
    model.load_state_dict(load['model_state_dict'])
    # model = torch.jit.load(os.path.join(PATH_TO_MODEL, 'model_quantized.pt'))
    model.to(DEVICE).eval()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():  # Capturing video
        ret, frame = cap.read()
        start = time.time()

        # Image preprocessing for format and shape required by model
        subframe_coords = get_subframe_coords(frame_w=frame.shape[1], frame_h=frame.shape[0],
                                              subframe_w=300, subframe_h=300)

        cl_image = frame[subframe_coords[1]:subframe_coords[3], subframe_coords[0]:subframe_coords[2]]
        cl_image = cv2.resize(cl_image, IMAGE_SIZE)
        cl_image = cv2.cvtColor(cl_image, cv2.COLOR_BGR2GRAY)

        frame = cv2.rectangle(frame,
                              (subframe_coords[0], subframe_coords[1]),
                              (subframe_coords[2], subframe_coords[3]),
                              color=(0, 255, 0),
                              thickness=2)
        frame = cv2.putText(frame,
                            "PLACE YOUR FACE IN THE BOX",
                            (subframe_coords[0], subframe_coords[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color=(255, 0, 0),
                            thickness=3)

        cl_image = ToTensor()(cl_image).unsqueeze(0).to(DEVICE)
        pred_emo = EMOTION_MAP[model(cl_image).argmax(dim=1).item()]

        frame = cv2.putText(frame,
                            pred_emo,
                            (subframe_coords[0], subframe_coords[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color=(0, 255, 0),
                            thickness=2)

        fps = 1. / (time.time() - start)
        # cv2.imshow('classification pipeline', cl_image[0, 0, ...].detach().numpy())
        cv2.imshow('classification pipeline', frame)

        print(fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to evaluate recognition model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_model', type=str, default='log/classification/20.03.30_20-13', help='path to model')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--emotions', type=str, default='Anger Happy Neutral Surprise',
                        help='emotions that model have trained on (space separated)\n')

    opt = parser.parse_args()
    # Initialising detection model
    PATH_TO_MODEL = os.path.join('..', opt.path_to_model)
    DEVICE = 'cpu'
    IMAGE_SIZE = (opt.image_size, opt.image_size)
    EMOTION_MAP = opt.emotions.split()

    run_eval()
