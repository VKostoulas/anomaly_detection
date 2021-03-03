import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Add, AveragePooling2D, UpSampling2D, InputLayer, \
    Activation, BatchNormalization, Conv2DTranspose
from functions.auto_encoder_functions import define_layers_sizes, \
    variational_latent_block, check_assertions


def advanced_auto_encoder(input_layer_shape, output_channels, filters_keep_percentage, min_filters, latent_dim,
                          kernel_initializer='glorot_uniform', filters='advanced_auto_encoder_default',
                          kernel_size=3, mode='ae', apply_batch_norm=False):
    """

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
    model_name = 'advanced_auto_encoder'
    check_assertions(model_name, apply_batch_norm, filters, mode)
    filters, latent_dim = define_layers_sizes(model_name, filters, latent_dim, filters_keep_percentage, min_filters)
    first_filters, enc_filters, dec_filters, final_filters = filters.values() if isinstance(filters, dict) else filters
    conv_args = {'strides': 1, 'padding': 'same', 'kernel_initializer': kernel_initializer}

    # First conv without batch normalization
    inputs = Input(shape=input_layer_shape)
    x = Conv2D(filters=first_filters[0], kernel_size=kernel_size, use_bias=not apply_batch_norm,
               name='initial_conv', **conv_args)(inputs)

    # Encoder
    for i, filter_size in enumerate(enc_filters):
        x = advanced_encode_step(input_volume=x, filters=filter_size, kernel_size=kernel_size,
                                 apply_batch_norm=apply_batch_norm, names=[f'down_{i + 1}', f'down_residual_{i + 1}'],
                                 conv_args=conv_args)

    # Latent space
    if mode == 'ae':
        # x = advanced_encode_step(input_volume=x, filters=latent_dim, kernel_size=kernel_size,
        #                          apply_batch_norm=apply_batch_norm, names=['latent', 'latent_residual'],
        #                          conv_args=conv_args)
        latent_block = tf.keras.Sequential([InputLayer(input_shape=x.shape[1:])], name='latent')
        latent_block.add(Conv2D(filters=latent_dim, kernel_size=kernel_size, use_bias=True, **conv_args))
        x = latent_block(x)
        # x = Conv2DTranspose(filters=dec_filters[0], kernel_size=kernel_size, use_bias=True, **conv_args)(x)

    elif mode == 'vae':
        x = variational_latent_block(latent_dim, x)

    # Decoder
    for i, filter_size in enumerate(dec_filters):
        x = advanced_decode_step(input_volume=x, filters=filter_size, kernel_size=kernel_size,
                                 apply_batch_norm=apply_batch_norm, names=[f'up_{i + 1}', f'up_residual_{i + 1}'],
                                 conv_args=conv_args)

    # Final pre-activation block
    # final_outputs = advanced_preactivation_block(input_volume=x, final_block_filters=final_filters[0],
    #                                              kernel_size=kernel_size, apply_batch_norm=apply_batch_norm,
    #                                              conv_args=conv_args)
    # TODO: kernel size 1, right?
    outputs = Conv2D(filters=output_channels, kernel_size=1, name='output', **conv_args)(x)
    outputs = Activation('tanh', dtype='float32', name='tanh')(outputs)

    return Model(inputs=inputs, outputs=outputs, name='advanced_auto_encoder')


def advanced_encode_step(input_volume, filters, kernel_size, apply_batch_norm, names, conv_args):
    """
    Down step for advanced_auto_encoder.

    :param input_volume:
    :param filters:
    :param kernel_size:
    :param apply_batch_norm:
    :param names:
    :param conv_args
    :return:
    """
    # TODO: Use bias should be the same for all here ?
    use_bias = not apply_batch_norm

    # Main path
    encode_block = tf.keras.Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=names[0])
    if apply_batch_norm:
        encode_block.add(BatchNormalization())
    encode_block.add(LeakyReLU(0.2))
    encode_block.add(AveragePooling2D(2))
    encode_block.add(Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))
    if apply_batch_norm:
        encode_block.add(BatchNormalization())
    encode_block.add(LeakyReLU(0.2))
    encode_block.add(Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))

    # Residual path
    residual_block = tf.keras.Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=names[1])
    residual_block.add(AveragePooling2D(2))
    # if the main path and residual path have not the same shapes, we add an 1x1 convolution
    if encode_block(input_volume).shape[1:] != residual_block(input_volume).shape[1:]:
        residual_block.add(Conv2D(filters=filters, kernel_size=1, use_bias=True, **conv_args))

    # Build paths and add them
    encode_path = encode_block(input_volume)
    residual_path = residual_block(input_volume)
    added_outputs = Add()([encode_path, residual_path])

    return added_outputs


def advanced_decode_step(input_volume, filters, kernel_size, apply_batch_norm, names, conv_args):
    """
    Up step for advanced_auto_encoder.

    :param input_volume:
    :param filters:
    :param kernel_size:
    :param apply_batch_norm:
    :param names:
    :param conv_args:
    :return:
    """
    # TODO: Use bias should be the same for all here ?
    use_bias = not apply_batch_norm

    decode_block = tf.keras.Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=names[0])
    if apply_batch_norm:
        decode_block.add(BatchNormalization())
    decode_block.add(LeakyReLU(0.2))
    decode_block.add(UpSampling2D(2))
    decode_block.add(Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))
    if apply_batch_norm:
        decode_block.add(BatchNormalization())
    decode_block.add(LeakyReLU(0.2))
    decode_block.add(Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))

    residual_block = tf.keras.Sequential([InputLayer(input_shape=input_volume.shape[1:])], name=names[1])
    residual_block.add(UpSampling2D(2))
    # if the main path and residual path have not the same shapes, we add an 1x1 convolution
    if decode_block(input_volume).shape[1:] != residual_block(input_volume).shape[1:]:
        residual_block.add(Conv2D(filters=filters, kernel_size=1, use_bias=use_bias, **conv_args))

    decode_path = decode_block(input_volume)
    residual_path = residual_block(input_volume)
    added_outputs = Add()([decode_path, residual_path])

    return added_outputs


def advanced_preactivation_block(input_volume, final_block_filters, kernel_size, apply_batch_norm, conv_args):
    """
    Final block of advanced autoencoder.

    :param input_volume:
    :param final_block_filters:
    :param kernel_size:
    :param apply_batch_norm:
    :param conv_args:
    :return:
    """
    # TODO: Use bias should be the same for all here ?
    use_bias = not apply_batch_norm

    # TODO: in the paper they say 'pre-activation residual block without normalization'. This means that they don't
    #       add batch normalization?
    final_block = tf.keras.Sequential([InputLayer(input_shape=input_volume.shape[1:])], name='final_block')
    if apply_batch_norm:
        final_block.add(BatchNormalization())
    final_block.add(LeakyReLU(0.2))

    final_block.add(Conv2D(filters=final_block_filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))
    if apply_batch_norm:
        final_block.add(BatchNormalization())
    final_block.add(LeakyReLU(0.2))

    # TODO: use_bias = True?
    final_block.add(Conv2D(filters=final_block_filters, kernel_size=kernel_size, use_bias=True, **conv_args))

    # TODO: after the final residual convolution, there is no batch normalization. This means that we should
    #       add use_bias=True?
    final_residual_block = tf.keras.Sequential([InputLayer(input_shape=input_volume.shape[1:])], name='final_residual')
    if final_block(input_volume).shape[1:] != final_residual_block(input_volume).shape[1:]:
        final_residual_block.add(Conv2D(filters=final_block_filters, kernel_size=1, use_bias=True, **conv_args))

    final_path = final_block(input_volume)
    final_residual_path = final_residual_block(input_volume)

    final_outputs = Add()([final_path, final_residual_path])

    return final_outputs