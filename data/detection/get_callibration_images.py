import cv2
import os
import datetime


PATH_TO_SAVE = 'D:/Emotion-Recognition-PRJCT2019/data/detection/callibration_images'
INTERNAL = False
device = 0 if INTERNAL else 1

cap = cv2.VideoCapture(device)
emotions = ['happy', 'surprised', 'angry', 'neutral', 'sad']

print('Video capture started..')
print(f'Please take 5 photos, each should capture one of the following emotions: {emotions}')
print('Press F key to make photo')
print('==========================')

i = 0
print(f'Stay \'{emotions[i]}\'')
while cap.isOpened():  # Capturing video
    ret, image = cap.read()
    cv2.imshow('flow', image)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        cv2.imwrite(os.path.join(PATH_TO_SAVE, datetime.datetime.now().strftime('%y%m%d%H%M%S') + '.jpg'), image)
        print(f'\'{emotions[i]}\' captured. {len(emotions) - i - 1} emotions left', end='\n\n')
        i += 1

        print(f'Stay \'{emotions[i]}\'')
