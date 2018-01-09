#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model


def cnn_model(input_shape=(224,224,3), num_classes=3):

    input_img = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Dropout(0.25, name='block1_dropout1')(x)
    
    # Block 2
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2,2), name='block2_pool1')(x)
    x = Dropout(0.25, name='block2_dropout1')(x)
    
    # Block 3
    x = Flatten(name='block3_flatten1')(x)
    x = Dense(512, activation='relu', name='block3_dense1')(x)
    x = Dropout(0.5, name='block3_dropout1')(x)
    x = Dense(num_classes, activation='softmax', name='block3_dense1')(x)

    cnn = Model(input_img, x)

    cnn.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    cnn.summary()

    return cnn


if __name__ == '__main__':
    model = cnn_model()
    