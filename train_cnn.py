import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import argparse
import tensorflow as tf
import time

np.random.seed(1234)

from src import CompactCNN, pipeline_train, pipeline_test

MEL_PATH = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Classify 2.")
    parser.add_argument('--mel_path', type=str, default='./melon/',
                        help='specify the directory where are stored mel-spectrogram and features')
    parser.add_argument('--active_multi_gpu', type=int, default=0, help='0: NO GPU, !=0 -> Multi Gpu')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--restore_epochs', type=int, default=0, help='Epoch From Which We Have to restoe')
    parser.add_argument('--num_images', type=int, default=11, help='Random Number of Images')
    parser.add_argument('--nb_conv_layers', type=int, default=4, help='Number of Conv. Layers')
    parser.add_argument('--n_verb_batch', type=int, default=10, help='Number of Batch to Print Verbose')
    parser.add_argument('--buffer_size', type=int, default=5, help='Buffer Size')

    return parser.parse_args()


def run():
    args = parse_args()

    #########################################################################################################
    # MODEL SETTING

    MEL_PATH = args.mel_path
    # if args.machine == 'server':
    #     MEL_PATH = '/home/daniele/Project/PreProcessing-MillionDatasetsPlaylist/original_dataset/hd/MPD-Extracted/arena_mel'
    # else:
    #     MEL_PATH = './original_dataset/mel/arena_mel'

    batch_size = args.batch_size
    lr = args.lr
    nb_conv_layers = args.nb_conv_layers
    saving_filepath = './training_weights_epoch_{0}/'

    if args.active_multi_gpu == 0:
        print('\n******\nDisable Multi-GPU\n******\n')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        physical_devices = ['cpu']
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy()
        print('\n******\nExecute in {0} Multi-GPU\n******\n'.format(len(physical_devices)))

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
        print('USE FULL DATA')
    else:
        num_images = args.num_images
        list_of_images = np.random.randint(0, num_all_images - 1, num_images)  # Random num_images indices
        print('USE RANDOM {0} DATA'.format(num_images))

    np.random.shuffle(list_of_images)
    train_ix = int(num_images * 0.9)
    train_indices = list_of_images[0:train_ix]
    num_train_samples = len(train_indices)
    test_indices = list_of_images[train_ix:]
    num_test_samples = len(test_indices)

    print('\n*********\nNum. Images {0}'.format(num_images))
    print('Num. Train Images {0}'.format(len(train_indices)))
    print('Num. Test Images {0}\n*********\n'.format(len(test_indices)))

    BUFFER_SIZE = args.buffer_size

    BATCH_SIZE_PER_REPLICA = args.batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    EPOCHS = args.epochs

    train_data = pipeline_train(MEL_PATH, train_indices, train_indices, BUFFER_SIZE, GLOBAL_BATCH_SIZE, EPOCHS)
    test_data = pipeline_test(MEL_PATH, test_indices, test_indices, GLOBAL_BATCH_SIZE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_data)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_data)

    # Create a checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # step_checkpoint_dir = './step_checkpoints'
    # step_checkpoint_prefix = os.path.join(step_checkpoint_dir, "ckpt")

    #########################################################################################################

    #########################################################################################################
    # Initialize Network

    with strategy.scope():
        cnn = CompactCNN(input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalization, dense_units,
                         output_shape, activation, dropout, args.batch_size, GLOBAL_BATCH_SIZE, strategy)

        # checkpoint = tf.train.Checkpoint(optimizer=cnn.optimizer, model=cnn.network)
        # step_checkpoint = tf.train.Checkpoint(optimizer=cnn.optimizer, model=cnn.network)

        # Restore
        if args.restore_epochs > 0:
            try:
                # checkpoint.restore(os.path.join(checkpoint_dir, 'ckpt-{}'.format(args.restore_epochs)))
                # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
                cnn.load_weights(saving_filepath.format(args.restore_epochs)).expect_partial()
                print('Model Successfully Restore at Epoch {}!'.format(args.restore_epochs))
            except Exception as ex:
                print('Model Do Not Restored!')
                print(ex)
                args.restore_epochs = 0

    print('Start Model Training for {0} Epochs!'.format(args.epochs - args.restore_epochs))
    total_batches = num_train_samples // batch_size + 1
    for epoch in range(EPOCHS - args.restore_epochs):
        start_epoch = time.time()
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0

        start = time.time()
        for idx, x in enumerate(train_dist_dataset):
            try:
                total_loss += cnn.distributed_train_step(x)
            except Exception as ex:
                print('\tERROR on Batch-id {}\n\t{}'.format(idx, ex))
            num_batches += 1
            if (idx + 1) % args.n_verb_batch == 0:
                print('\rEpoch %d/%d - %d/%d - %.3f sec/it' % (
                    epoch + args.restore_epochs + 1, EPOCHS, idx + 1, total_batches // len(physical_devices),
                    (time.time() - start) / args.n_verb_batch))
                start = time.time()

            # if (idx % 5000 == 0) and (idx != 0):
            #     # This Checkpoint Can Be Useful in the Case of an Error Stopping after 10K steps in an epoch
            #     # We need to implement a custom restore if it will happen a lot of times.
            #     step_checkpoint.save(step_checkpoint_prefix)
            #     print(
            #         '------> Backup Checkpoint Saved in {} at Step {} of the Epoch {}'.format(step_checkpoint_dir, idx,
            #                                                                                   epoch + 1))

        train_loss = total_loss / num_batches

        #########################################################################################################
        # SAVE
        print('\nModel-Weights Saving...')
        cnn.save_weights(saving_filepath.format(epoch+1), overwrite=True, save_format=None)
        # checkpoint.save(checkpoint_prefix)
        print('Model-Weights Saved At Epoch {}'.format(args.restore_epochs + epoch + 1))

        # TEST LOOP
        for x in test_dist_dataset:
            cnn.distributed_test_step(x)

        template = ("\n\t\tEpoch %d/%d, Loss: %.3f, Accuracy: %.3f, "
                    "Test Accuracy: %.3f in %.2f sec\n")
        print(template % (epoch + args.restore_epochs + 1, args.epochs, train_loss,
                          cnn.train_accuracy.result() * 100,
                          cnn.test_accuracy.result() * 100, (time.time() - start_epoch)))

        cnn.train_accuracy.reset_states()
        cnn.test_accuracy.reset_states()

        start_epoch = time.time()

    #########################################################################################################

    # #########################################################################################################
    # # SAVE
    # print('\nModel Weights Saving at the End of the Training...')
    # cnn.save_weights(saving_filepath.format(args.epochs), overwrite=True, save_format=None)
    # print('..Model Weights Saved')

    #########################################################################################################
    # TEST
    # print('\nModel Evaluation')
    #
    # average_accuracy = 0.0
    # num_steps_test = num_test_samples // batch_size + 1
    # for x in test_dist_dataset:
    #     average_accuracy += cnn.distributed_test_step(x)
    #
    # print('Accuracy on test set: %.3f' % ((average_accuracy / num_steps_test) * 100))


if __name__ == '__main__':
    run()
