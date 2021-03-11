from src import CompactCNN, load_spectrograms

MEL_PATH = 'original_dataset/mel/arena_mel'

load_spectrograms(MEL_PATH, range(100))
test_data = None
input_shape = test_data[0,:,:,:].shape

# number of Convolutional Layers
nb_conv_layers = 4

# number of Filters in each layer
nb_filters = [128, 384, 768, 2048]

lr = 0.001

n_mels = input_shape[0]

normalization = 'batch'

# number of hidden layers at the end of the model
nb_hidden = 0
dense_units = 200

# IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
output_shape = 64
# output_shape = item_vecs_reg.shape[1]

# Output activation
activation = 'linear'

dropout = 0

CompactCNN(input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalization, nb_hidden, dense_units, output_shape, activation, dropout)