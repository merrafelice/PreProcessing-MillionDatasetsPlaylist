import json
import pandas as pd
import numpy as np
import os


with open('original_dataset/song_meta.json') as json_file:
    data = json.load(json_file)
    classes = {}
    class_num = 0
    for i, song in enumerate(data):
        if i % 1000 == 0:
            print(i)
        one_hot = np.zeros(30)  # 30 Genres
        for genre in song['song_gn_gnr_basket']:
            if genre not in classes:
                classes[genre] = class_num
                class_num += 1
            one_hot[classes[genre]] = 1
        np.save('melon/classes/{0}.npy'.format(song['id']), one_hot)