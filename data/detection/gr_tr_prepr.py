"""metadata_orig.txt => metadata.txt (ground truth)"""
import json

with open('metadata_orig.txt', 'r') as file:
    txt = json.load(file)

with open('metadata.txt', 'w') as file:
    arr = []
    for i in txt:
        arr.append({
            'fileName': i['fileName'][0],
            'facialEmotion': i['facialEmotion'],
            'faceRect': i['faceRect']
        })
    json.dump(arr, file, indent=True)
