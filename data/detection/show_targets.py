import cv2
import os
import json
from utils.utils import xywh2xyxy


def show_target_images(images_path='images/',
                       markup_path='data_markup.txt',
                       color=(0, 0, 0),
                       thickness=5,
                       max_images=5):

    markup = json.load(open(markup_path, 'r'))
    for image_name in os.listdir(images_path)[0:max_images]:
        name = image_name[:16]
        points = xywh2xyxy(markup[name])
        img = cv2.imread(images_path + image_name)
        show_rectangle(img, points, name=name, color=color, thickness=thickness)


def show_rectangle(image, rectangular, name='image', color=(0, 0, 0), thickness=5):
    img = cv2.rectangle(image, (rectangular[0], rectangular[1]), (rectangular[2], rectangular[3]), color, thickness)
    cv2.imshow(name, img)
    cv2.waitKey(0)
