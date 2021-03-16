import json
import pandas as pd

# Playlist-Tracks File Structure
data_matrix = pd.DataFrame(columns=['playlist_id', 'song_id', 'feedback', 'timestamp'])
data_matrix['playlist_id'] = data_matrix['playlist_id'].astype(int)

# Side-Information Structure
## Playlist
playlist_title = pd.DataFrame(columns=['playlist_id', 'title'])
playlist_like_cnt = pd.DataFrame(columns=['playlist_id', 'like_cnt'])
playlist_tag = pd.DataFrame(columns=['playlist_id', 'tag'])

## Track
track_title = pd.DataFrame(columns=['song_id', 'title'])
track_album = pd.DataFrame(columns=['song_id', 'album_id'])
track_artist = pd.DataFrame(columns=['song_id', 'artist_id'])
track_genre_detailed = pd.DataFrame(columns=['song_id', 'genre_detailed'])
track_genre = pd.DataFrame(columns=['song_id', 'genre'])

### Genre Name
genre_title = pd.DataFrame(columns=['genre_id', 'title'])


def load_data_into_pd(filename):
    global data_matrix
    global playlist_title
    global playlist_like_cnt
    global playlist_tag

    with open('original_dataset/{0}.json'.format(filename)) as json_file:
        data = json.load(json_file)
        for playlist in data:
            for relative_timestamp, song in enumerate(playlist['songs']):
                data_matrix = data_matrix.append({
                    'playlist_id': playlist['id'],
                    'song_id': song,
                    'feedback': 1,
                    'timestamp': relative_timestamp
                }, ignore_index=True)

            playlist_title = playlist_title.append(
                {'playlist_id': playlist['id'],
                 'title': playlist['plylst_title']
                 }, ignore_index=True
            )

            playlist_like_cnt = playlist_like_cnt.append(
                {'playlist_id': playlist['id'],
                 'like_cnt': playlist['like_cnt']
                 }, ignore_index=True
            )

            for tag in playlist['tags']:
                playlist_tag = playlist_tag.append(
                    {'playlist_id': playlist['id'],
                     'tag': tag
                     }, ignore_index=True
                )


print('Start Train')
load_data_into_pd('train')
print('Start Val')
load_data_into_pd('val')
print('Start Test')
load_data_into_pd('test')

with open('original_dataset/song_meta.json') as json_file:
    data = json.load(json_file)
    for i, song in enumerate(data):
        if i % 1000 == 0:
            print(i)
        track_title = track_title.append({
            'song_id': song['id'],
            'title': song['song_name']
        }, ignore_index=True)

        track_album = track_album.append({
            'song_id': song['id'],
            'album_id': song['album_id']
        }, ignore_index=True)

        for artist in song['artist_id_basket']:
            track_artist = track_artist.append({
                'song_id': song['id'],
                'artist_id': artist
            }, ignore_index=True)

        for genre_detailed in song['song_gn_dtl_gnr_basket']:
            track_genre_detailed = track_genre_detailed.append({
                'song_id': song['id'],
                'genre_detailed': genre_detailed
            }, ignore_index=True)

        for genre in song['song_gn_gnr_basket']:
            track_genre = track_genre.append({
                'song_id': song['id'],
                'genre': genre
            }, ignore_index=True)
import numpy as np

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


# Load Genre
with open('original_dataset/genre_gn_all.json') as json_file:
    data = json.load(json_file)
    for genre_id, title in data.items():
        genre_title = genre_title.append({
            'genre_id': genre_id,
            'title': title
        }, ignore_index=True)

# Store
import os

os.makedirs('melon')

path_name = 'melon/{0}.tsv'

data_matrix.to_csv(path_name.format('dataset'), sep='\t', index=None, header=None)
playlist_title.to_csv(path_name.format('playlist_title'), sep='\t', index=None, header=None)
playlist_like_cnt.to_csv(path_name.format('playlist_like_cnt'), sep='\t', index=None, header=None)
playlist_tag.to_csv(path_name.format('playlist_tag'), sep='\t', index=None, header=None)
track_title.to_csv(path_name.format('song_title'), sep='\t', index=None, header=None)
track_album.to_csv(path_name.format('track_album'), sep='\t', index=None, header=None)
track_artist.to_csv(path_name.format('track_artist'), sep='\t', index=None, header=None)
track_genre_detailed.to_csv(path_name.format('track_genre_detailed'), sep='\t', index=None, header=None)
track_genre.to_csv(path_name.format('track_genre'), sep='\t', index=None, header=None)
genre_title.to_csv(path_name.format('genre_title'), sep='\t', index=None, header=None)
