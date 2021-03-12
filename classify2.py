import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src import CompactCNN, load_spectrograms, pipeline_train, pipeline_test

MEL_PATH = '/home/daniele/Project/PreProcessing-MillionDatasetsPlaylist/original_dataset/hd/MPD-Extracted/arena_mel'

# load_spectrograms(MEL_PATH, range(100))
# test_data = None

batch_size = 16
epochs = 2

dir_list = os.listdir(MEL_PATH)
num_dir = [int(d) for d in dir_list]
last_dir = max(num_dir)
num_images = max([int(d.split('.')[0]) for d in os.listdir(os.path.join(MEL_PATH, str(last_dir)))]) + 1

list_of_images = np.arange(num_images)
print('Num. Images {0}'.format(num_images))

np.random.shuffle(list_of_images)
train_ix = (num_images*0.9)
train_indices = list_of_images[0:train_ix]
num_train_samples = len(train_indices)
test_indices = list_of_images[train_ix:]
num_test_samples = len(test_indices)

train_data = pipeline_train(train_indices, train_indices, batch_size, epochs)
test_data = pipeline_test(test_indices, test_indices, batch_size)

# number of Convolutional Layers
nb_conv_layers = 4

# number of Filters in each layer
nb_filters = [128, 384, 768, 2048]

lr = 0.001

n_mels = 48
input_shape = (48, 1876, 1)

normalization = 'batch'

# number of hidden layers at the end of the model
nb_hidden = 0
dense_units = 200

# IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
output_shape = 30
# output_shape = item_vecs_reg.shape[1]

# Output activation
activation = 'linear'

dropout = 0

cnn = CompactCNN(input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalization, nb_hidden, dense_units, output_shape, activation, dropout)

num_steps = num_train_samples // batch_size + 1
count_steps = 0
average_loss = 0.0

# training
for idx, d in enumerate(train_data):
    average_loss += cnn.train_step(d)
    if count_steps == num_steps:
        print('\n\n******************************************')
        print('Epoch is over!')
        print('Average loss: %f' % (average_loss / num_steps))
        count_steps = 0
        average_loss = 0.0
    else:
        count_steps += 1

    if (idx + 1) % 100 == 0:
        sys.stdout.write('\r%d/%d samples completed' % (idx + 1, num_steps))
        sys.stdout.flush()

average_accuracy = 0.0
num_steps_test = num_test_samples // batch_size + 1

# test
for d in test_data:
    average_accuracy += cnn.predict_on_batch(d)

print('Accuracy on test set: %f' % (average_accuracy / num_steps_test))
