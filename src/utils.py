import numpy as np
import os
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa

import tensorflow as tf


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
    S = data
    plt.figure(figsize=(12, 8))
    # D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # plt.subplot(4, 2, 1)
    librosa.display.specshow(S, sr=16000, y_axis='mel', x_axis='time', hop_length=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.show()
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


def add_channel(data, n_channels=1):
    # n_channels: 1 for grey-scale, 3 for RGB, but usually already present in the data

    N, ydim, xdim = data.shape
    data = data.reshape(N, ydim, xdim, n_channels)

    return data