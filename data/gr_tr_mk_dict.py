"""metadata.txt => dict_metadata.txt (ground truth)"""
import json

with open('metadata.txt', 'r') as file:
    txt = json.load(file)
    d = {}
    for i in txt:
        d[i['fileName'][:-1]] = i['faceRect']

    json.dump(d, open('dict_metadata.txt', 'w'), indent=True)
