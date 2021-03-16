from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Convolution2D, LayerNormalization, ELU, MaxPooling2D, \
    Flatten, Dropout, Dense

from src.utils import get_pooling


class CompactCNN(tf.keras.Model):

    def __init__(self, input_shape, lr, nb_conv_layers, nb_filters, n_mels, normalize, dense_units,
                 output_shape, activation, dropout, batch_size, *args, **kwargs):

        super(CompactCNN, self).__init__()
        self._input_shape = input_shape
        self._lr = lr
        self._nb_conv_layers = nb_conv_layers
        self._n_mels = n_mels
        self._normalize = normalize
        self._dropout = dropout
        self._output_shape = output_shape
        self._activation = activation
        self._dense_units = dense_units
        self._poolings = get_pooling(self._n_mels)
        self._channel_axis = 1  # 3?
        self._batch_size = batch_size

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

        for dense_unit in self._dense_units:
            self.network.add(Dropout(self._dropout))
            self.network.add(Dense(dense_unit, activation='relu'))

        # Output Layer
        self.network.add(Dense(self._output_shape, activation='sigmoid'))

        # Tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # self.network.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.accuracy)

    #@tf.function
    def call(self, inputs, training=None, mask=None):
        predicted = self.network(inputs, training)
        return predicted

    #@tf.function
    def compile(self):
        super(CompactCNN, self).compile()
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self._lr)
        # Loss
        # self.loss = tf.keras.losses.BinaryCrossentropy()
        self.loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # Metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy()

    #@tf.function
    def train_step(self, batch):
        song, genre = batch
        with tf.GradientTape() as t:
            predicted = self(song, training=True)
            # loss = self.loss(genre, predicted)
            loss = tf.reduce_sum(self.loss(genre, predicted)) * (1. / self._batch_size)
        grads = t.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.accuracy.update_state(genre, predicted)
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "BinaryAccuracy": self.accuracy.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.accuracy]

    #@tf.function
    def predict_on_batch(self, batch):
        x, y_true = batch
        y_pred = self(x, training=False)
        self.accuracy.update_state(y_true, y_pred)

        return self.accuracy.result()
