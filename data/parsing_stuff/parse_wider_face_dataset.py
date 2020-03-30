import scipy.io
import numpy
import os
import shutil
import json


def convert_coords(coords):
    # [x_lt, y_lt, w, h] -> [x_center, y_center, w, h]
    coords[0] = coords[0] + coords[2] // 2
    coords[1] = coords[1] + coords[3] // 2
    return coords


PATH_TO_MARKUP = 'D:\Emotion-Recognition-PRJCT2019\data\detection\wider_face_train.mat'
raw_data = scipy.io.loadmat(PATH_TO_MARKUP)
data = {}
# [blur_label_list, expression_label_list, face_bbx_list, file_list, illumination_label_list,
# invalid_label_list, occlusion_label_list, pose_label_list]

for i in range(61):
    outer_data = []
    for _, key in enumerate(raw_data.keys()):
        inner_data = []
        if _ < 3 or _ == 4:
            pass
        else:
            for elem in raw_data[key][i][0]:
                if key == 'face_bbx_list':
                    inner_data.append(elem[0])
                elif key == 'file_list':
                    inner_data.append(elem[0][0])
                else:
                    try:
                        inner_data.append(elem[0][0][0][0])
                    except IndexError:
                        inner_data.append(elem[0][0][0])
            outer_data.append(inner_data)
    data[str(i)] = numpy.array(outer_data).transpose()

# print(len(data))
markup = {}
for key in data:
    new_key = data[key][0][3].split('_')[0]
    markup[new_key] = []
    for elem in data[key]:
        if len(elem[2]) > 1 or elem[0] == 2 or elem[4] == 1 or elem[5] == 1 or elem[6] == 2 or elem[7] == 1:
            pass
        else:
            markup[new_key].append((elem[3], elem[2].tolist()))


for sub_dir in os.listdir('C:/Users/Stasek/Downloads/WIDER_train/images'):
    key = sub_dir.split('--')[0]
    for img_name, _ in markup[key]:
        shutil.copyfile(os.path.join('C:/Users/Stasek/Downloads/WIDER_train/images', sub_dir, (img_name + '.jpg')),
                        os.path.join('C:/Users/Stasek/Downloads/WIDER_train', 'tesst', (img_name + '.jpg')))


with open('D:\Emotion-Recognition-PRJCT2019\data\detection\data_markup.txt', 'r') as file:
    old_markup = json.load(file)

new_markup = {}
temp_markup = {}
for key in markup:
    for elem in markup[key]:
        temp_markup[elem[0]] = elem[1][0]

for filename in os.listdir('C:/Users/Stasek/Downloads/WIDER_train/tesst'):
    new_markup[filename[:-4]] = convert_coords(temp_markup[filename[:-4]])

for filename in os.listdir('D:/Emotion-Recognition-PRJCT2019/data/detection/train_images'):
    new_markup[filename[:-4]] = old_markup[filename[:16]]


for filename in os.listdir('C:/Users/Stasek/Downloads/WIDER_train/tesst'):
    shutil.copy(os.path.join('C:/Users/Stasek/Downloads/WIDER_train/tesst', filename),
                os.path.join('D:/test_dir/train_images', filename))


json.dump(new_markup, open('D:/test_dir/train_markup.txt', 'w'), indent=True)
