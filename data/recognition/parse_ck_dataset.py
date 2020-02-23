import os
from PIL import Image
import shutil

# PATH_TO_DATA = 'D:/emodata/CK+48'
#
# for emo_dir in os.listdir(PATH_TO_DATA):
#     for i, img in enumerate(os.listdir(os.path.join(PATH_TO_DATA, emo_dir))):
#         if i % 3 == 0:
#             pil_img = Image.open(os.path.join(PATH_TO_DATA, emo_dir, img))
#             new_pil_img = pil_img.resize((64, 64))
#             try:
#                 new_pil_img.save(os.path.join('D:/emodata/emorec_v2', emo_dir, img))
#             except FileNotFoundError:
#                 os.mkdir(os.path.join('D:/emodata/emorec_v2', emo_dir))
#                 new_pil_img.save(os.path.join('D:/emodata/emorec_v2', emo_dir, img))

import pandas as pd

data = pd.read_csv('D:/emodata/fer2013.csv')

print(type(data['pixels'][0]))


