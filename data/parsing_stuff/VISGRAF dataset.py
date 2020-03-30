import os
import shutil

path = "../../Downloads/facesdb 2"


class_names = {"Disgust": "05_img.bmp", "Surprise": "03_img.bmp", "Anger": "04_img.bmp", "Neutral": "00_img.bmp"}

for class_name in class_names:
    os.makedirs(os.path.join("recognition", class_name), exist_ok=True)

folders = [i for i in os.listdir(path) if not i.startswith(".")]

for folder in folders:
    source_dir = os.path.join(path, folder, "bmp")
    for file_name in os.listdir(source_dir):
        for class_name in class_names.keys():
            if file_name.endswith(class_names[class_name]):
                dest_dir = os.path.join("data/recognition", class_name)
                print(dest_dir)
                print(shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name)))

