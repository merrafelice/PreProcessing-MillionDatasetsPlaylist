import os
import sys
import numpy as np
import argparse
import tensorflow as tf

np.random.seed(1234)

from src import CompactCNN, pipeline_train, pipeline_test

MEL_PATH = '/home/daniele/Project/PreProcessing-MillionDatasetsPlaylist/original_dataset/hd/MPD-Extracted/arena_mel'

strategy = tf.distribute.MirroredStrategy()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Classify 2.")
    parser.add_argument('--machine', type=str, default='home', help='help or server')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--restore', type=int, default=0, help='Restore Model')

    return parser.parse_args()



def run():
    args = parse_args()

    #########################################################################################################
    # MODEL SETTING

    if args.machine == 'server':
        MEL_PATH = '/home/daniele/Project/PreProcessing-MillionDatasetsPlaylist/original_dataset/hd/MPD-Extracted/arena_mel'
    else:
        MEL_PATH = './original_dataset/mel/arena_mel'

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    nb_conv_layers = 4

    # number of Filters in each layer
    nb_filters = [128, 384, 768, 2048]
    n_mels = 48
    input_shape = (48, 1876, 1)
    normalization = 'batch'
    # number of hidden layers at the end of the model
    dense_units = [200]
    output_shape = 30
    # Output activation
    activation = 'linear'
    dropout = 0

    saving_filepath = './checkpoint/e:{0}_bs:{1}_lr:{2}'.format(args.epochs, args.batch_size, args.lr)
    #########################################################################################################

    #########################################################################################################
    # READ DATA with pipeline
    dir_list = os.listdir(MEL_PATH)
    num_dir = [int(d) for d in dir_list]
    last_dir = max(num_dir)
    num_images = max([int(d.split('.')[0]) for d in os.listdir(os.path.join(MEL_PATH, str(last_dir)))]) + 1

    list_of_images = np.arange(num_images)
    print('Num. Images {0}'.format(num_images))

    np.random.shuffle(list_of_images)
    train_ix = int(num_images * 0.9)
    train_indices = list_of_images[0:train_ix]
    num_train_samples = len(train_indices)
    test_indices = list_of_images[train_ix:]
    num_test_samples = len(test_indices)

    BUFFER_SIZE = num_train_samples

    BATCH_SIZE_PER_REPLICA = args.batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    EPOCHS = args.epochs

    train_data = pipeline_train(MEL_PATH, train_indices, train_indices, BUFFER_SIZE, GLOBAL_BATCH_SIZE, EPOCHS)
    test_data = pipeline_test(MEL_PATH, test_indices, test_indices, GLOBAL_BATCH_SIZE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_data)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_data)

    #########################################################################################################

    #########################################################################################################
    # Initialize Network

    with strategy.scope():
        cnn = CompactCNN(input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalization, dense_units,
                     output_shape, activation, dropout, args.batch_size, GLOBAL_BATCH_SIZE, strategy)

    # Restore
    if args.restore == 1:
        cnn.load_weights(saving_filepath).expect_partial()
        print('Model Successfully Restore!')
    else:
        print('Start Model Training for {0} Epochs!'.format(args.epochs))

        num_steps = num_train_samples // batch_size + 1
        count_steps = 0
        average_loss, average_acc = 0.0, 0.0
        count_epochs = 1

        # Training of the Network
        for idx, d in enumerate(train_dist_dataset):
            l = cnn.distributed_train_step(d)
            average_loss += l
            average_acc += a
            if count_steps == num_steps:
                print('\n\n******************************************')
                print('Epoch is over!')
                print('Average loss: %f' % (average_loss / num_steps))
                print('******************************************')
                count_steps, average_loss, average_acc = 0, 0.0, 0.0
                count_epochs += 1
            else:
                count_steps += 1

            if (idx + 1) % 1000 == 0:
                sys.stdout.write('\rEpoch %d - %d/%d samples completed - Loss: %.3f' % (count_epochs, (idx + 1) % num_steps, num_steps))
                sys.stdout.flush()
                break

    #########################################################################################################

    #########################################################################################################
    # SAVE
    cnn.save_weights(saving_filepath, overwrite=True, save_format=None)

    #########################################################################################################
    # TEST
    print('\nModel Evaluation...')

    average_accuracy = 0.0
    num_steps_test = num_test_samples // batch_size + 1
    for x in test_dist_dataset:
        average_accuracy += cnn.distributed_test_step(x)

    print('Accuracy on test set: %f' % (average_accuracy / num_steps_test))




if __name__ == '__main__':
    run()
