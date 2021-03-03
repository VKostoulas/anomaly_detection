"""
Functional autoencoder
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Conv2DTranspose, InputLayer, Activation, \
    BatchNormalization, ReLU
from functions.auto_encoder_functions import define_layers_sizes, \
    variational_latent_block, check_assertions


def simple_auto_encoder(input_layer_shape, output_channels, filters_keep_percentage, min_filters, latent_dim=1024,
                        kernel_initializer='glorot_uniform', filters='functional_auto_encoder_default',
                        kernel_size=2, mode='ae', apply_batch_norm=False):
    """
    Autoencoder created with functional and sequential API.

    :param input_layer_shape:
    :param output_channels:
    :param filters_keep_percentage:
    :param min_filters:
    :param latent_dim:
    :param kernel_initializer:
    :param filters:
    :type filters: any
    :param kernel_size:
    :param mode:
    :param apply_batch_norm:
    :return:
    """
    model_name = 'functional_auto_encoder'
    check_assertions(model_name, apply_batch_norm, filters, mode)
    filters, latent_dim = define_layers_sizes(model_name, filters, latent_dim, filters_keep_percentage, min_filters)
    _, encoder_filters, decoder_filters = filters.values() if isinstance(filters, dict) else filters
    conv_args = {'strides': 2, 'kernel_size': kernel_size, 'padding': 'same', 'kernel_initializer': kernel_initializer,
                 'use_bias': not apply_batch_norm}

    inputs = Input(shape=input_layer_shape)
    x = inputs

    # Remaining layers of Encoder
    for i, _f in enumerate(encoder_filters):
        x = simple_encode_step(input_volume=x, filters=_f, conv_args=conv_args, name='down{}'.format(i))

    # Latent space
    x = simple_latent_block(input_volume=x, latent_dim=latent_dim, conv_args=conv_args, mode=mode)

    # Decoder
    for i, _f in enumerate(decoder_filters):
        x = simple_decode_step(input_volume=x, filters=_f, conv_args=conv_args, name='up{}'.format(i))

    # Last layer of Decoder
    model_output = Conv2DTranspose(filters=output_channels, name='final_conv', **conv_args)(x)
    model_output = Activation('tanh', dtype='float32', name='tanh')(model_output)

    return tf.keras.Model(inputs=inputs, outputs=model_output, name='functional_auto_encoder')


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
    if not conv_args['use_bias']:
        block.add(BatchNormalization())
    block.add(LeakyReLU(alpha=0.2))
    return block(input_volume)


def simple_latent_block(input_volume, latent_dim, conv_args, mode):
    """

    :param input_volume:
    :param latent_dim:
    :param apply_batch_norm:
    :param conv_args:
    :param mode:
    :return:
    """
    if mode == 'ae':
        # TODO: is this ok ?
        x = simple_encode_step(input_volume=input_volume, filters=latent_dim, conv_args=conv_args, name='latent')

    elif mode == 'vae':
        x = variational_latent_block(latent_dim, input_volume)
    return x


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
    if not conv_args['use_bias']:
        block.add(BatchNormalization())
    block.add(ReLU())
    return block(input_volume)
