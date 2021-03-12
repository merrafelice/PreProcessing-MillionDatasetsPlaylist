import json
import pandas as pd
import numpy as np
import os

print('Extract Features')
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
print('End Extract Features')

print('Extract Data')
with open('melon/data.tsv', 'w') as output_file:
    for filename in ['train', 'val', 'test']:
        with open('original_dataset/{0}.json'.format(filename)) as json_file:
            data = json.load(json_file)
            for i, playlist in enumerate(data):
                if i % 1000 == 0:
                    print(i)
                for relative_timestamp, song in enumerate(playlist['songs']):
                    output_file.write("{0}\t{1}\t{2}\t{3}\n".format(playlist['id'], song, 1, relative_timestamp))
print('END Extract Data')


