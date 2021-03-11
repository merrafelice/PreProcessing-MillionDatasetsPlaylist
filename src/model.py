from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Convolution2D, LayerNormalization, ELU, MaxPooling2D, \
    Flatten, Dropout, Dense

from src.utils import get_pooling


class CompactCNN(tf.keras.Model, ABC):

    def __init__(self, input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalize, nb_hidden, dense_units,
                 output_shape, activation, dropout):

        self._input_shape = input_shape
        self._lr = lr
        self._nb_conv_layers = nb_conv_layers
        self._nb_hidden = nb_hidden
        self._n_mels = n_mels
        self._normalize = normalize
        self._dropout = dropout
        self._output_shape = output_shape
        self._activation = activation
        self._dense_units = dense_units
        self._poolings = get_pooling(self._n_mels)
        self._channel_axis = 1  # 3?

        self.network = tf.keras.Sequential()
        # Input block
        self.network.add(Input(shape=self._input_shape))
        self.network.add(BatchNormalization(axis=self._channel_axis, name='bn_0_freq'))

        if self._normalize == 'batch':
            pass
        elif self._normalize in ('data_sample', 'time', 'freq', 'channel'):
            self.network.add(LayerNormalization(self._normalize, name='nomalization'))
        elif self._normalize in ('no', 'False'):
            # Manage x = melgram_input
            pass

        for index in range(self._nb_conv_layers):
            # Conv block 1
            self.network.add(Convolution2D(nb_filters[index], (3, 3), padding='same'))
            self.network.add(BatchNormalization(axis=self._channel_axis))
            self.network.add(ELU())
            self.network.add(MaxPooling2D(pool_size=self._poolings[index]))

        # Flatten the outout of the last Conv Layer
        self.network.add(Flatten())

        for index in range(self._nb_hidden):
            self.network.add(Dropout(self._dropout))
            self.network.add(Dense(self._dense_units[index], activation='relu'))

        # Output Layer
        self.network.add(Dense(self._output_shape, activation=self._activation))

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self._lr)

        # advanced:
        # self.optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

        # Loss
        self.loss = tf.keras.losses.CategoricalCrossentropy()


    @tf.function
    def call(self, inputs, training=None, mask=None):
        x, l = inputs
        pass

    @tf.function
    def train_step(self, batch):
        pass
