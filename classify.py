import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from src import CompactCNN, load_spectrograms, pipeline_train, pipeline_test

MEL_PATH = 'original_dataset/mel/arena_mel'

# load_spectrograms(MEL_PATH, range(100))
# test_data = None

batch_size = 16
epochs = 2

train_x_list = os.listdir('original_dataset/songs/train/')
train_x_list.sort(key=lambda x: int(x.split(".")[0]))
train_x_list = [int(f.split('.')[0]) for f in train_x_list]
train_y_list = os.listdir('original_dataset/genres/train/')
train_y_list.sort(key=lambda x: int(x.split(".")[0]))
train_y_list = [int(f.split('.')[0]) for f in train_y_list]

test_x_list = os.listdir('original_dataset/songs/test/')
test_x_list.sort(key=lambda x: int(x.split(".")[0]))
test_x_list = [int(f.split('.')[0]) for f in test_x_list]
test_y_list = os.listdir('original_dataset/genres/test/')
test_y_list.sort(key=lambda x: int(x.split(".")[0]))
test_y_list = [int(f.split('.')[0]) for f in test_y_list]

num_train_samples = len(train_x_list)
num_test_samples = len(test_x_list)

train_data = pipeline_train(train_x_list, train_y_list, batch_size, epochs)
test_data = pipeline_test(test_x_list, test_y_list, batch_size)

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
