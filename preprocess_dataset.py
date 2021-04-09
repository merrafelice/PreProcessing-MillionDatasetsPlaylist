import json
import pandas as pd
import numpy as np
import os

# print('Extract Features')
# with open('original_dataset/song_meta.json') as json_file:
#     data = json.load(json_file)
#     classes = {}
#     class_num = 0
#     for i, song in enumerate(data):
#         if i % 1000 == 0:
#             print(i)
#         one_hot = np.zeros(30)  # 30 Genres
#         for genre in song['song_gn_gnr_basket']:
#             if genre not in classes:
#                 classes[genre] = class_num
#                 class_num += 1
#             one_hot[classes[genre]] = 1
#         np.save('melon/classes/{0}.npy'.format(song['id']), one_hot)
# print('End Extract Features')

print('Extract Data')
dict_tmp = dict()
with open('melon/data.tsv', 'w') as output_file:
    for filename in ['train', 'val', 'test']:
        with open('original_dataset/{0}.json'.format(filename)) as json_file:
            data = json.load(json_file)
            for i, playlist in enumerate(data):
                if i % 1000 == 0:
                    print(i)
                for song in playlist['songs']:
                    if playlist['id'] not in dict_tmp:
                        dict_tmp[playlist['id']] = -1
                    relative_timestamp = dict_tmp[playlist['id']] + 1
                    output_file.write("{0}\t{1}\t{2}\t{3}\n".format(playlist['id'], song, 1, relative_timestamp))
                    dict_tmp[playlist['id']] = relative_timestamp
print('END Extract Data')

import pandas as pd

data = pd.read_csv('melon/data.tsv', sep='\t', header=None)

test = pd.DataFrame({'original': data[0].unique(), 'after': data[0].unique()})
test = test.sort_values(by='original')

test.to_csv('melon/visual_feats.tsv', header=None, sep='\t', index=None)
print('End')

print('Time Splitting')
import datetime

years = [0, 2015, 2020]  # 0 is the minimum placeholder
for i, year in enumerate(years):
    if year != 0:
        print('Extract Data for {0}'.format(year))
        dict_tmp = dict()
        with open('melon/data_{0}.tsv'.format(year), 'w') as output_file:
            for filename in ['train', 'val', 'test']:
                with open('original_dataset/{0}.json'.format(filename)) as json_file:
                    data = json.load(json_file)
                    j = 0
                    for playlist in data:
                        date_time_obj = datetime.datetime.strptime(playlist['updt_date'], '%Y-%m-%d %H:%M:%S.%f')
                        if years[i - 1] < date_time_obj.year <= years[i]:
                            j += 1
                            if j % 1000 == 0:
                                print(j)
                            for song in playlist['songs']:
                                if playlist['id'] not in dict_tmp:
                                    dict_tmp[playlist['id']] = -1
                                relative_timestamp = dict_tmp[playlist['id']] + 1
                                output_file.write(
                                    "{0}\t{1}\t{2}\t{3}\n".format(playlist['id'], song, 1, relative_timestamp))
                                dict_tmp[playlist['id']] = relative_timestamp
        print('End Extract Data for {0}'.format(year))
