
from functions.auto_encoder_functions import variational_latent_block

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, MaxPool2D, Input, InputLayer, LeakyReLU, \
    ReLU, Activation, UpSampling2D, AveragePooling2D


def inception_step(input_volume, filters, conv_args, reduce_dims, name):
    """

    :param input_volume:
    :param filters:
    :param conv_args:
    :param reduce_dims:
    :param name:
    :return:
    """
    filters_1x1, filters_2x2_reduce, filters_2x2, filters_4x4_reduce, filters_4x4, filters_pool_prod = filters
    block1 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_1x1')
    block2 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_2x2')
    block3 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_4x4')
    block4 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_pool')

    block1.add(Conv2D(filters_1x1, kernel_size=(1, 1), strides=1, **conv_args))
    block4.add(AveragePooling2D(pool_size=(2, 2), strides=1, padding='same'))

    if reduce_dims:
        block2.add(Conv2D(filters_2x2_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block3.add(Conv2D(filters_4x4_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block4.add(Conv2D(filters_pool_prod, kernel_size=(1, 1), strides=1, **conv_args))

    block2.add(Conv2D(filters_2x2, kernel_size=(2, 2), strides=1, **conv_args))
    block3.add(Conv2D(filters_4x4, kernel_size=(4, 4), strides=1, **conv_args))

    layer_outputs = [block1(input_volume), block2(input_volume), block3(input_volume), block4(input_volume)]
    concat = Concatenate(axis=-1, name=name + '_output')(layer_outputs)

    return concat


def inception_encode_step(input_volume, filters, conv_args, reduce_dims, name):
    """

    :param input_volume:
    :param filters:
    :param conv_args:
    :param reduce_dims:
    :param name:
    :return:
    """
    filters_1x1, filters_2x2_reduce, filters_2x2, filters_4x4_reduce, filters_4x4, filters_pool_prod = filters
    block1 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_1x1')
    block2 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_2x2')
    block3 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_4x4')
    block4 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_pool')

    block1.add(Conv2D(filters_1x1, kernel_size=(1, 1), strides=2, **conv_args))
    block4.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

    if reduce_dims:
        # block2.add(Conv2D(filters_2x2_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block3.add(Conv2D(filters_4x4_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block4.add(Conv2D(filters_pool_prod, kernel_size=(1, 1), strides=1, **conv_args))

    block2.add(Conv2D(filters_2x2, kernel_size=(2, 2), strides=2, **conv_args))
    block3.add(Conv2D(filters_4x4, kernel_size=(4, 4), strides=2, **conv_args))

    layer_outputs = [block1(input_volume), block2(input_volume), block3(input_volume), block4(input_volume)]
    concat = Concatenate(axis=-1, name=name + '_output')(layer_outputs)

    return concat


def inception_decode_step(input_volume, filters, conv_args, reduce_dims, name):
    """

    :param input_volume:
    :param filters:
    :param conv_args:
    :param reduce_dims:
    :param name:
    :return:
    """

    filters_1x1, filters_2x2_reduce, filters_2x2, filters_4x4_reduce, filters_4x4, filters_pool_prod = filters
    block1 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_1x1')
    block2 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_2x2')
    block3 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_4x4')
    block4 = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_pool')

    block1.add(Conv2DTranspose(filters_1x1, kernel_size=(1, 1), strides=2, **conv_args))
    block4.add(UpSampling2D())

    if reduce_dims:
        # block2.add(Conv2D(filters_2x2_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block3.add(Conv2D(filters_4x4_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block4.add(Conv2D(filters_pool_prod, kernel_size=(1, 1), strides=1, **conv_args))

    block2.add(Conv2DTranspose(filters_2x2, kernel_size=(2, 2), strides=2, **conv_args))
    block3.add(Conv2DTranspose(filters_4x4, kernel_size=(4, 4), strides=2, **conv_args))

    layer_outputs = [block1(input_volume), block2(input_volume), block3(input_volume), block4(input_volume)]
    concat = Concatenate(axis=-1, name=name + '_output')(layer_outputs)

    return concat


def simple_encode_step(input_volume, filters, conv_args, name):
    """

    :param input_volume:
    :param filters:
    :param apply_batch_norm:
    :param conv_args:
    :param name:
    :return:
    """

    block = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name)
    block.add(Conv2D(filters=filters, **conv_args))
    block.add(LeakyReLU(alpha=0.2))
    return block(input_volume)


def simple_decode_step(input_volume, filters, conv_args, name):
    """

    :param input_volume:
    :param filters:
    :param apply_batch_norm:
    :param conv_args:
    :param name:
    :param mode:
    :return:
    """

    block = Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=name)
    block.add(Conv2DTranspose(filters=filters, **conv_args))
    block.add(ReLU())
    return block(input_volume)


# def inception_auto_encoder(input_layer_shape, output_channels, filters_keep_percentage=1.0, min_filters=16,
#                            latent_dim=1024, kernel_initializer='glorot_uniform'):
#
#     inputs = Input(shape=input_layer_shape)
#     inc_conv_args = {'kernel_initializer': kernel_initializer, 'use_bias': True, 'padding': 'same',
#                      'activation': 'relu'}
#     ae_conv_args = {'strides': 2, 'kernel_size': 2, 'padding': 'same', 'kernel_initializer': kernel_initializer}
#
#     x = inception_step(inputs, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=False, name='inc_block_1')
#     x = simple_encode_step(input_volume=x, filters=64, conv_args=ae_conv_args, name='down_block_1')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_2')
#     x = simple_encode_step(input_volume=x, filters=128, conv_args=ae_conv_args, name='down_block_2')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_3')
#     x = simple_encode_step(input_volume=x, filters=128, conv_args=ae_conv_args, name='down_block_3')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_4')
#     x = simple_encode_step(input_volume=x, filters=256, conv_args=ae_conv_args, name='down_block_4')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_5')
#
#     # Latent
#     x = simple_encode_step(input_volume=x, filters=latent_dim, conv_args=ae_conv_args, name='latent')
#
#     x = simple_decode_step(input_volume=x, filters=256, conv_args=ae_conv_args, name='up_block_1')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_6')
#     x = simple_decode_step(input_volume=x, filters=128, conv_args=ae_conv_args, name='up_block_2')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_7')
#     x = simple_decode_step(input_volume=x, filters=128, conv_args=ae_conv_args, name='up_block_3')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_8')
#     x = simple_decode_step(input_volume=x, filters=64, conv_args=ae_conv_args, name='up_block_4')
#     x = inception_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_block_9')
#
#     model_output = Conv2DTranspose(filters=output_channels, name='final_conv', **ae_conv_args)(x)
#     model_output = Activation('tanh', dtype='float32', name='tanh')(model_output)
#
#     return Model(inputs=inputs, outputs=model_output, name='inception_auto_encoder')


def inception_auto_encoder(input_layer_shape, output_channels, filters_keep_percentage=1.0, min_filters=16,
                           latent_dim=1024, kernel_initializer='glorot_uniform'):

    inputs = Input(shape=input_layer_shape)
    inc_conv_args = {'kernel_initializer': kernel_initializer, 'use_bias': True, 'padding': 'same',
                     'activation': 'relu'}
    ae_conv_args = {'strides': 2, 'kernel_size': 2, 'padding': 'same', 'kernel_initializer': kernel_initializer}

    x = inception_encode_step(inputs, [16, 96, 32, 16, 16, 16], inc_conv_args, reduce_dims=True, name='inc_enc_block_1')
    x = inception_encode_step(x, [32, 96, 64, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_enc_block_2')
    x = inception_encode_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_enc_block_3')
    x = inception_encode_step(x, [128, 128, 256, 32, 64, 64], inc_conv_args, reduce_dims=True, name='inc_enc_block_4')

    # Latent
    x = inception_encode_step(x, [64, 128, 256, 64, 64, 64], inc_conv_args, reduce_dims=True, name='latent')

    x = inception_decode_step(x, [128, 128, 256, 32, 64, 64], inc_conv_args, reduce_dims=True, name='inc_dec_block_5')
    x = inception_decode_step(x, [64, 96, 128, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_dec_block_6')
    x = inception_decode_step(x, [32, 96, 64, 16, 32, 32], inc_conv_args, reduce_dims=True, name='inc_dec_block_7')
    x = inception_decode_step(x, [16, 96, 32, 16, 16, 16], inc_conv_args, reduce_dims=True, name='inc_dec_block_8')

    x = Conv2DTranspose(filters=output_channels, name='final_conv', **ae_conv_args)(x)
    model_output = Activation('tanh', dtype='float32', name='tanh')(x)

    return Model(inputs=inputs, outputs=model_output, name='inception_auto_encoder')
