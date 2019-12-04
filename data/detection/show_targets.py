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
        show_rectangles(img, [points], name=name, color=color, thickness=thickness)


def show_rectangles(image, rectangles, confs, name='image', color=(0, 255, 0), thickness=2):
    img = image
    for rectangle, conf in zip(rectangles, confs):
        img = cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), color, thickness)
        img = cv2.putText(img, str(conf), (rectangle[0], rectangle[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness)
    cv2.imshow(name, img)
    cv2.waitKey(0)
