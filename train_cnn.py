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
    parser.add_argument('--epochs', type=int, default=2, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--restore', type=int, default=0, help='Restore Model')
    parser.add_argument('--num_images', type=int, default=100, help='Random Number of Images')
    parser.add_argument('--nb_conv_layers', type=int, default=4, help='Number of Conv. Layers')
    parser.add_argument('--n_verb_batch', type=int, default=10, help='Number of Batch to Print Verbose')

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

    if args.active_multi_gpu == 0:
        print('\n******\nDisable Multi-GPU\n******\n')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    else:
        print('\n******\nExecute in Multi-GPU\n******\n')
        strategy = tf.distribute.MirroredStrategy()

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
    num_all_images = max([int(d.split('.')[0]) for d in os.listdir(os.path.join(os.path.join(MEL_PATH, 'arena_mel'), str(last_dir)))]) + 1

    if args.num_images == -1:
        num_images = num_all_images
        list_of_images = np.arange(num_all_images)  # All the Images are stored from 0 to N-1
        print('USE FULL DATA')
    else:
        num_images = args.num_images
        list_of_images = np.random.randint(0, num_all_images-1, num_images)  # Random num_images indices
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

    BUFFER_SIZE = num_train_samples

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
    #########################################################################################################

    #########################################################################################################
    # Initialize Network

    with strategy.scope():
        cnn = CompactCNN(input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalization, dense_units,
                         output_shape, activation, dropout, args.batch_size, GLOBAL_BATCH_SIZE, strategy)

        checkpoint = tf.train.Checkpoint(optimizer=cnn.optimizer, model=cnn.network)

        # Restore
        if args.restore == 1:
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            # cnn.load_weights(saving_filepath).expect_partial()
            print('Model Successfully Restore!')

    if args.restore != 1:
        print('Start Model Training for {0} Epochs!'.format(args.epochs))
        total_batches = num_train_samples // batch_size + 1
        for epoch in range(EPOCHS):
            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0

            start = time.time()
            for idx, x in enumerate(train_dist_dataset):
                total_loss += cnn.distributed_train_step(x)
                num_batches += 1
                if (idx + 1) % args.n_verb_batch == 0:
                    sys.stdout.write('\rEpoch %d - %d/%d - %.3f sec/it' % (
                        epoch + 1, idx + 1, total_batches, (time.time() - start) / args.n_verb_batch))
                    sys.stdout.flush()
                    start = time.time()

            train_loss = total_loss / num_batches

            # TEST LOOP
            for x in test_dist_dataset:
                cnn.distributed_test_step(x)

            if epoch % 2 == 0:
                checkpoint.save(checkpoint_prefix)

            template = ("\nEpoch %d, Loss: %.3f, Accuracy: %.3f, "
                        "Test Accuracy: %.3f")
            print(template % (epoch + 1, train_loss,
                              cnn.train_accuracy.result() * 100,
                              cnn.test_accuracy.result() * 100))

            cnn.train_accuracy.reset_states()
            cnn.test_accuracy.reset_states()

            # num_steps = num_train_samples // batch_size + 1
            # count_steps = 0
            # average_loss, average_acc = 0.0, 0.0
            # count_epochs = 1
            #
            # # Training of the Network
            # for idx, d in enumerate(train_dist_dataset):
            #     l, a = cnn.distributed_train_step(d)
            #     average_loss += l
            #     average_acc += a
            #     if count_steps == num_steps:
            #         print('\n******************************************')
            #         print('Epoch {0} is over!')
            #         print('Average loss: %f' % (average_loss / num_steps))
            #         print('******************************************\n')
            #         checkpoint.save(checkpoint_prefix)
            #         count_steps, average_loss, average_acc = 1, 0.0, 0.0
            #         count_epochs += 1
            #     else:
            #         count_steps += 1
            #
            #     if (idx + 1) % 10 == 0:
            #         sys.stdout.write('\rEpoch %d - %d/%d samples completed - Loss: %.3f - Acc: %.3f' % (
            #         count_epochs, (idx + 1) % num_steps, num_steps, average_loss / count_steps, average_acc / count_steps))
            #         sys.stdout.flush()

    #########################################################################################################

    #########################################################################################################
    # SAVE
    # print('\nModel Saving...')
    # cnn.save_weights(saving_filepath, overwrite=True, save_format=None)
    # print('Model Saved...')

    #########################################################################################################
    # TEST
    print('\nModel Evaluation...')

    average_accuracy = 0.0
    num_steps_test = num_test_samples // batch_size + 1
    for x in test_dist_dataset:
        average_accuracy += cnn.distributed_test_step(x)

    print('Accuracy on test set: %.3f' % ((average_accuracy / num_steps_test) * 100))


if __name__ == '__main__':
    run()
