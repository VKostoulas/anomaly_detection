import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dense, Dropout, InputLayer, \
    GlobalAveragePooling2D, Activation, Conv2DTranspose, ReLU, Layer, Reshape, Flatten


def simple_classifier(input_shape, num_of_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(kernel_size=3, filters=64, strides=2, padding='same'),
        # BatchNormalization(),
        LeakyReLU(0.2),
        Conv2D(kernel_size=3, filters=128, strides=2, padding='same'),
        # BatchNormalization(),
        LeakyReLU(0.2),
        Conv2D(kernel_size=3, filters=256, strides=2, padding='same'),
        # BatchNormalization(),
        LeakyReLU(0.2),
        Conv2D(kernel_size=3, filters=512, strides=2, padding='same'),
        # BatchNormalization(),
        LeakyReLU(0.2),

        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.4),
        Dense(num_of_classes),
        Activation('softmax', dtype='float32', name='softmax')
    ])
    return model

