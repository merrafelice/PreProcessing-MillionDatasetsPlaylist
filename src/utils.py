import numpy as np
import os

import pickle
import tensorflow as tf

MEL_PATH = '/home/daniele/Project/PreProcessing-MillionDatasetsPlaylist/original_dataset/hd/MPD-Extracted/arena_mel'


def get_pooling(n_mels):
    if n_mels >= 256:
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif n_mels >= 128:
        poolings = [(2, 4), (4, 5), (4, 8), (4, 7), (4, 4)]
    elif n_mels >= 96:
        poolings = [(2, 4), (4, 5), (3, 8), (4, 7), (4, 3)]  # (2, 8), (4, 3)]
    elif n_mels >= 72:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
    elif n_mels >= 64:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 48:
        poolings = [(2, 4), (4, 5), (3, 8), (2, 7), (4, 4)]
    elif n_mels >= 32:
        poolings = [(2, 4), (2, 5), (3, 8), (2, 7), (4, 4)]
    elif n_mels >= 24:
        poolings = [(2, 4), (2, 4), (3, 8), (2, 8), (4, 4)]
    elif n_mels >= 16:
        poolings = [(2, 4), (2, 5), (2, 8), (2, 7), (4, 4)]
    elif n_mels >= 8:
        poolings = [(2, 4), (2, 4), (2, 8), (1, 8), (4, 4)]
    return poolings


def show_spectrograms(ind, data):
    import librosa.display
    import librosa
    import matplotlib.pyplot as plt

    S = data
    plt.figure(figsize=(12, 8))
    # D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # plt.subplot(4, 2, 1)
    librosa.display.specshow(S, sr=16000, y_axis='mel', x_axis='time', hop_length=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    # plt.show()
    # from PIL import Image
    # fig = plt.Figure()
    # canvas = FigureCanvas(fig)
    # ax = fig.add_subplot(111)
    # # librosa.amplitude_to_db(data, ref=np.max)
    # p = librosa.display.specshow(data=data, sr=16, ax=ax, y_axis='mel', x_axis='time', hop_length=512, cmap=cm.jet)
    # fig.savefig('spec.png')

    # for mode in ['1', 'L', 'P', 'RGB', 'RGBA']:
    #     img = Image.fromarray(data, mode=mode)
    #     img.show()


def load_spectrograms(mel_path, item_ids, enc=True):
    list_spectrograms = []
    ret_ids = []
    for p, kid in enumerate(item_ids):
        directory_number = kid // 1000
        npz_spec_file = '{0}/{1}/{2}.npy'.format(mel_path, directory_number, kid)
        if os.path.exists(npz_spec_file):
            melspec = np.load(npz_spec_file)
            if melspec.shape[1] < 1876:
                print(melspec.shape)
            else:
                list_spectrograms.append(melspec[:, :1876])
                ret_ids.append(p)
        else:
            print("File not exists", npz_spec_file)
    item_list = np.array(list_spectrograms)
    item_list[np.isinf(item_list)] = 0
    # for ind, item in enumerate(item_list):
    #     show_spectrograms(ind, item)
    item_list = add_channel(item_list)

    return item_list, ret_ids


def load_func(s, g):
    song = np.expand_dims(np.load('{0}arena_mel/{1}/{2}.npy'.format(MEL_PATH, s.numpy() // 1000, str(s.numpy()))), -1)
    if song.shape != (48, 1876, 1):
        song = tf.image.resize(song, [48, 1876])
    genre = np.load('{0}classes/{1}.npy'.format(MEL_PATH, str(g.numpy())))
    return song, genre


def load_func_extract(s):
    song = np.expand_dims(np.load('{0}arena_mel/{1}/{2}.npy'.format(MEL_PATH, s.numpy() // 1000, str(s.numpy()))), -1)
    if song.shape != (48, 1876, 1):
        song = tf.image.resize(song, [48, 1876])
    return song, s


def load_func_train(s, g):
    song = np.expand_dims(np.load('./original_dataset/songs/train/' + str(s.numpy()) + '.npy'), -1)
    genre = np.load('./original_dataset/genres/train/' + str(g.numpy()) + '.npy')
    return song, genre


def load_func_test(s, g):
    song = np.expand_dims(np.load('./original_dataset/songs/test/' + str(s.numpy()) + '.npy'), -1)
    genre = np.load('./original_dataset/genres/test/' + str(g.numpy()) + '.npy')
    return song, genre


def pipeline_train(mel_path, songs, genres, BUFFER_SIZE, GLOBAL_BATCH_SIZE, EPOCHS):
    def load_wrapper(s, g):
        o = tf.py_function(
            load_func,
            (s, g,),
            (np.float32, np.int8)
        )
        return o

    global MEL_PATH
    MEL_PATH = mel_path
    data = tf.data.Dataset.from_tensor_slices((songs, genres))
    data = data.map(load_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 3 sec
    data = data.shuffle(buffer_size=BUFFER_SIZE, seed=1234, reshuffle_each_iteration=True)
    # data = data.repeat(EPOCHS)
    data = data.batch(batch_size=GLOBAL_BATCH_SIZE)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    # Test
    # data = data.shuffle(BUFFER_SIZE, seed=1234, reshuffle_each_iteration=True).batch(GLOBAL_BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data


def pipeline_extract_features(mel_path, songs, genres, BUFFER_SIZE, GLOBAL_BATCH_SIZE, EPOCHS):
    def load_wrapper(s):
        o = tf.py_function(
            load_func_extract,
            (s,),
            (np.float32, np.int64)
        )
        return o

    global MEL_PATH
    MEL_PATH = mel_path
    data = tf.data.Dataset.from_tensor_slices((songs))
    data = data.map(load_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # data = data.shuffle(buffer_size=100, seed=1234, reshuffle_each_iteration=True)
    # data = data.repeat(EPOCHS)
    # data = data.batch(batch_size=batch_size)
    # data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    data = data.batch(GLOBAL_BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


def pipeline_test(mel_path, songs, genres, GLOBAL_BATCH_SIZE):
    def load_wrapper(s, g):
        o = tf.py_function(
            load_func,
            (s, g,),
            (np.float32, np.int8)
        )
        return o

    global MEL_PATH
    MEL_PATH = mel_path
    data = tf.data.Dataset.from_tensor_slices((songs, genres))
    data = data.map(load_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # data = data.batch(batch_size=batch_size)
    # data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    data = data.batch(GLOBAL_BATCH_SIZE)
    return data


def add_channel(data, n_channels=1):
    # n_channels: 1 for grey-scale, 3 for RGB, but usually already present in the data

    N, ydim, xdim = data.shape
    data = data.reshape(N, ydim, xdim, n_channels)

    return data


def restore_weights(cnn, saving_filepath):
    try:
        with open(saving_filepath, "rb") as f:
            cnn.set_model_state(pickle.load(f))
        print(f"Model correctly Restored")

    except Exception as ex:
        print(f"Error in model restoring operation! {ex}")

    return False
