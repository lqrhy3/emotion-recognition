import os
import pandas as pd
import shutil


emotion_map = ['Anger', 'Surprise', 'Neutral', 'Sad', 'Happy', 'Disgust']

PATH_TO_DATA = 'data/images'
labels = pd.read_csv('data/recognition/legend.csv')
labels['emotion'] = labels['emotion'].apply(lambda name: name.lower().title())
labels['emotion'] = labels['emotion'].map({'Happiness': 'Happy',
                                           'Sadness': 'Sad', 'Anger': 'Anger', 'Surprise': 'Surprise',
                                           'Disgust': 'Disgust', 'Neutral': 'Neutral'})
labels.dropna(inplace=True)

for line in range(len(labels)):
    name = labels.iloc[[line]]['image'].values[0]
    emotion = labels.iloc[[line]]['emotion'].values[0]
    # print(labels.iloc[[line]]['image'].values[0], labels.iloc[[line]]['emotion'].values[0])
    shutil.copy(os.path.join(PATH_TO_DATA, name), os.path.join('data/fer', emotion, name))
