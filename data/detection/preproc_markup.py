import json

with open('metadata_original.txt', 'r') as file:
    txt = json.load(file)


def clean_markup():
    """(Markup) metadata_original.txt -> metadata.txt
    From all markup data leave only {"fileName": , "facialEmotion": , 'faceRect': }
    """
    with open('metadata.txt', 'w') as file:
        arr = []
        for i in txt:
            arr.append({
                'fileName': i['fileName'][0],
                'facialEmotion': i['facialEmotion'],
                'faceRect': list(map(int, i['faceRect']))
            })
        json.dump(arr, file, indent=True)


def prepoc_markup():
    """(Markup) metadata.txt -> data_markup.txt
    Structure enhancment and coordinates converting
    """
    with open('metadata.txt', 'r') as file:
        txt = json.load(file)
        d = {}
        for i in txt:
            d[i['fileName'][:-1]] = facerect_preproc(i['faceRect'])

        json.dump(d, open('data_markup.txt', 'w'), indent=True)


def facerect_preproc(coord):
    """Convert face rectangle coordinates
    Args:
        coord: [x_lt, y_lt, w, h]
    Returns:
        [x_center, y_center, w, h]
            """
    coord[0] = coord[0] + coord[2] // 2
    coord[1] = coord[1] + coord[3] // 2
    return coord
