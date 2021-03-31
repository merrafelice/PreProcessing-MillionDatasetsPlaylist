import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import argparse
import tensorflow as tf
import time

np.random.seed(1234)

from src import CompactCNN, pipeline_extract_features

MEL_PATH = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Classify 2.")
    parser.add_argument('--mel_path', type=str, default='./melon/',
                        help='specify the directory where are stored mel-spectrogram and features')
    parser.add_argument('--active_gpu', type=int, default=-1, help='-1: NO GPU, 1 Gpu-ID')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--restore_epochs', type=int, default=10, help='Epoch From Which We Have to restoe')
    parser.add_argument('--num_images', type=int, default=101, help='Random Number of Images')
    parser.add_argument('--nb_conv_layers', type=int, default=4, help='Number of Conv. Layers')
    parser.add_argument('--n_verb_batch', type=int, default=10, help='Number of Batch to Print Verbose')
    parser.add_argument('--buffer_size', type=int, default=100, help='Buffer Size')

    return parser.parse_args()


def run():
    args = parse_args()

    #########################################################################################################
    # MODEL SETTING

    MEL_PATH = args.mel_path

    batch_size = args.batch_size
    lr = args.lr
    nb_conv_layers = args.nb_conv_layers

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.active_gpu)

    # number of Filters in each layer
    nb_filters = [128, 384, 768, 2048]
    n_mels = 48
    input_shape = (48, 1876, 1)
    normalization = 'batch'

    # number of hidden layers at the end of the model
    dense_units = []
    output_shape = 30
    # Output activation
    activation = 'linear'
    dropout = 0

    #########################################################################################################

    #########################################################################################################
    # READ DATA with pipeline
    dir_list = os.listdir(os.path.join(MEL_PATH, 'arena_mel'))
    num_dir = [int(d) for d in dir_list]
    last_dir = max(num_dir)
    num_all_images = max([int(d.split('.')[0]) for d in
                          os.listdir(os.path.join(os.path.join(MEL_PATH, 'arena_mel'), str(last_dir)))]) + 1

    if args.num_images == -1:
        num_images = num_all_images
        list_of_images = np.arange(num_all_images)  # All the Images are stored from 0 to N-1
        print('EXTRACT FULL SONGS')
    else:
        num_images = args.num_images
        list_of_images = np.arange(num_images - 1)  # Random num_images indices
        print('EXTRACT FIRST {0} SONGS'.format(num_images))

    print('\n*********\nNum. Images {0}'.format(num_images))

    BUFFER_SIZE = args.buffer_size

    BATCH_SIZE_PER_REPLICA = args.batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA

    EPOCHS = args.epochs

    data = pipeline_extract_features(MEL_PATH, list_of_images, list_of_images, BUFFER_SIZE, GLOBAL_BATCH_SIZE, EPOCHS)

    # Create a checkpoint directory to store the checkpoints.
    saving_filepath = './training_weights_epoch_{0}/'

    #########################################################################################################

    #########################################################################################################
    # Initialize Network

    cnn = CompactCNN(input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalization, dense_units,
                     output_shape, activation, dropout, args.batch_size, GLOBAL_BATCH_SIZE, None)

    cnn.load_weights(saving_filepath.format(args.restore_epochs)).expect_partial()
    print('Model Successfully Restore at Epoch {}!'.format(args.restore_epochs))

    # Create
    dir_fc = '{0}original/fully_connected'.format(MEL_PATH)
    dir_fm = '{0}original/feature_maps'.format(MEL_PATH)
    if os.path.exists(dir_fc):
        shutil.rmtree(dir_fc)
    os.makedirs(dir_fc)
    if os.path.exists(dir_fm):
        shutil.rmtree(dir_fm)
    os.makedirs(dir_fm)

    start = time.time()
    for idx, batch in enumerate(data):
        song, song_id = batch
        fcs = cnn.extract_feature(batch, 'flatten')
        fms = cnn.extract_feature(batch, 'elu_2')
        for song_in_batch_id, sid in enumerate(song_id):
            np.save('{}/{}.npy'.format(dir_fc, sid.numpy()), fcs[song_in_batch_id])
            np.save('{}/{}.npy'.format(dir_fm, sid.numpy()), fms[song_in_batch_id])

        if (idx + 1) % 10 == 0:
            print('Features Extracted for %d/%d Images in %.3f sec' % ((idx + 1)*args.batch_size, num_images, (time.time() - start)))
            start = time.time()


if __name__ == '__main__':
    run()
