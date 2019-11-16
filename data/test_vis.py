import cv2
import os
import json

color = (0, 0, 0)
thickness = 5

ground_tr = json.load(open('dict_metadata.txt', 'r'))
for filename in os.listdir('images')[350:380]:
    name = filename[:16]
    points = list(map(int, ground_tr[name]))
    im = cv2.imread('images/' + filename)
    image = cv2.rectangle(im, (points[0], points[1]), (points[0] + points[2], points[1] + points[3]), color, thickness)
    cv2.imshow('image', image)
    cv2.waitKey(0)
