import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, BatchNormalization, Reshape, \
    Conv2DTranspose, Add, Activation


# Helper function to apply activation and batch normalization to the
# output added with output of residual connection from the encoder

def lrelu_bn(inputs):
    lrelu = LeakyReLU()(inputs)
    bn = BatchNormalization()(lrelu)
    return bn


def unet_autoencoder(input_layer_shape, output_channels, kernel_size):
    # Input
    input_img = Input(shape=input_layer_shape)

    # Encoder
    y = Conv2D(32, kernel_size, padding='same', strides=(2, 2))(input_img)
    y = LeakyReLU()(y)
    y = Conv2D(64, kernel_size, padding='same', strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y1 = Conv2D(128, kernel_size, padding='same', strides=(2, 2))(y)           # skip-1
    y = LeakyReLU()(y1)
    y = Conv2D(256, kernel_size, padding='same', strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y2 = Conv2D(512, kernel_size, padding='same',strides=(2, 2))(y)           # skip-2
    y = LeakyReLU()(y2)
    y = Conv2D(512, kernel_size, padding='same', strides=(2, 2))(y)
    y = LeakyReLU()(y)
    # y = Conv2D(1024, kernel_size, padding='same', strides=(2, 2))(y)
    # y = LeakyReLU()(y)

    # Flattening for the bottleneck
    # vol = y.shape
    # x = Flatten()(y)
    # latent = Dense(128, activation='relu')(x)
    #
    # y = Dense(np.prod(vol[1:]), activation='relu')(latent)
    # y = Reshape((vol[1], vol[2], vol[3]))(y)
    # y = Conv2DTranspose(1024, kernel_size, padding='same')(y)
    # y = LeakyReLU()(y)
    y = Conv2DTranspose(512, kernel_size, padding='same', strides=(2, 2))(y)
    y = Add()([y2, y])                                                         # second skip connection added here
    y = lrelu_bn(y)
    y = Conv2DTranspose(256, kernel_size, padding='same', strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, kernel_size, padding='same', strides=(2, 2))(y)
    y = Add()([y1, y])                                                          # first skip connection added here
    y = lrelu_bn(y)
    y = Conv2DTranspose(128, kernel_size, padding='same', strides=(2, 2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(64, kernel_size, padding='same', strides=(2, 2))(y)
    y = LeakyReLU()(y)
    # y = Conv2DTranspose(32, kernel_size, padding='same', strides=(2, 2))(y)
    # y = LeakyReLU()(y)
    y = Conv2DTranspose(output_channels, kernel_size, padding='same', strides=(2, 2))(y)
    outputs = Activation('tanh', dtype='float32', name='tanh')(y)

    return Model(inputs=input_img, outputs=outputs, name='unet_auto_encoder')
