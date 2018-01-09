#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model


def vgg19_model(input_shape=(224,224,3), num_classes=3):

    input_img = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool1')(x)
    x = Dropout(0.25, name='block1_dropout1')(x)

    # Block 2
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2,2), name='block2_pool2')(x)
    x = Dropout(0.25, name='block2_dropout1')(x)

    # Block 3
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool1')(x)

    # Block 4
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool1')(x)

    # Block 5
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool1')(x)

    # Block6
    x = Flatten(name='block6_flatten1')(x)
    x = Dense(4096, activation='relu', name='block6_dense1')(x)
    x = Dense(4096, activation='relu', name='block6_dense2')(x)
    x = Dense(num_classes, activation='softmax', name='block6_dense3')(x)

    vgg19 = Model(input_img, x)

    vgg19.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    vgg19.summary()

    return vgg19


if __name__ == '__main__':
    model = vgg19_model()
    