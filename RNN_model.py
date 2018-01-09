#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Embedding, LSTM
from keras.models import Model


def rnn_model(input_shape=(10, 3)):
    hidden_neurons = input_shape[0]

    input_layer = Input(shape=input_shape, name='input1')

    x = LSTM(hidden_neurons, dropout=0.2, recurrent_dropout=0.2, name='LSTM1')(input_layer)
    x = Dense(1, activation='sigmoid', name='dense1')(x)

    rnn = Model(input_layer, x)

    rnn.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    rnn.summary()

    return rnn


if __name__ == '__main__':
    model = rnn_model()
    