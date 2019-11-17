"""metadata.txt => dict_metadata.txt (ground truth)"""
import json


def facerect_preprocc(arr):
    arr[0] = arr[0] + arr[2]//2
    arr[1] = arr[1] + arr[3]//2
    return arr


with open('metadata.txt', 'r') as file:
    txt = json.load(file)
    d = {}
    for i in txt:
        d[i['fileName'][:-1]] = facerect_preprocc(i['faceRect'])

    json.dump(d, open('dict_metadata.txt', 'w'), indent=True)


