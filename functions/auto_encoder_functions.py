import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import LeakyReLU, Dense, InputLayer, Flatten, Reshape
from tensorflow.keras.models import Sequential


class CustomLayers(tf.keras.layers.Layer):
    """
    Stack layers of a network.
    """

    def __init__(self, name, mode=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mode = self.define_preprocess_mode(mode)  # defines the preprocessing mode of classifier

    def get_config(self):
        """
        :return:
        :rtype:
        """
        config = super().get_config()
        return config

    @staticmethod
    def define_preprocess_mode(mode):

        if mode in ['MobileNet', 'InceptionV3', 'NASNetMobile', 'NASNetLarge', 'InceptionResNetV2', 'Resnet50']:
            mode = 'tf'
        elif mode in ['VGG16', 'VGG19']:
            mode = 'caffe'
        elif mode in []:
            mode = 'torch'
        elif mode in ['EfficientNetB0']:
            mode = 'no_scale'
        else:
            if mode:
                raise ValueError(f'Mode {mode} of classifier is not implemented.')
            else:
                pass
        return mode

    def call(self, x, training=None):  # pylint: disable=unused-argument, no-self-use, arguments-differ
        """
        Stack layers of a network so they constitute as one tensor.

        :param x: list of tensors
        :type x: list
        :param training: option to determine whether the model is in training phase or not
        :type training: bool
        :return: one tensor containing all items of x stacked
        :rtype: tf.tensor
        """
        if self.name == 'stack_layer':
            x = tf.stack(x)
        elif self.name == 'concat_layer':
            x = tf.concat(x, axis=0)
        elif self.name == 'variational_latent_block':
            x = tf.cast(tf.split(x, num_or_size_splits=2, axis=1), tf.float32)  # split to mean, logvar
            eps = tf.random.normal(shape=tf.shape(x[0]))  # sample a random normal number
            x = eps * tf.exp(x[1] * .5) + x[0]  # reparametrize
        elif self.name == 'rescale_and_preprocess_layer':
            # x is a double batch, containing both the images and their reconstructions
            # x is always normalized to -1, 1
            x += 1.
            x *= 127.5
            if self.mode == 'no_scale':
                pass
            else:
                x = preprocess_input(x, data_format='channels_last', mode=self.mode)
        else:
            raise ValueError(f"value of 'name' argument does not exist (value: {self.name}).")

        return x


def define_layers_sizes(model_name, filters, latent_dim, filters_keep_percentage, min_filters):
    """
    Function to define the current filters for FunctionalAutoEncoder

    :param model_name:
    :param filters:
    :param latent_dim:
    :param filters_keep_percentage:
    :param min_filters:
    :return:
    """
    if model_name == 'functional_auto_encoder':

        if filters == 'functional_auto_encoder_default':
            filters = {'first_conv_filters': [32],
                       'encoder_filters': [64, 128, 256, 512],
                       'decoder_filters': [512, 256, 128, 64, 32]}

        elif filters == 'functional_auto_encoder_variational':
            filters = {'first_conv_filters': [32],
                       'encoder_filters': [32, 32, 32],
                       'decoder_filters': [32, 32, 32]}

    elif model_name == 'custom_unet':

        if filters == 'custom_unet_default':
            filters = {'first_conv_filters': [32],
                       'encoder_filters': [64, 128, 256, 512],
                       'decoder_filters': [256, 128, 64, 32, 16]}

    elif model_name == 'advanced_auto_encoder':

        if filters == 'advanced_auto_encoder_default':
            filters = {'first_conv_filters': [64],
                       'encoder_filters': [32, 32, 32, 32, 32],
                       'decoder_filters': [32, 32, 32, 32, 32, 32],
                       'final_block_filters': [64]}

        elif filters == 'advanced_auto_encoder_variational':
            filters = {'first_conv_filters': [64],
                       'encoder_filters': [32, 32, 32, 32, 32],
                       'decoder_filters': [32, 32, 32, 32, 32],
                       'final_block_filters': [64]}

    new_filters = ()
    if isinstance(filters, dict):
        for filters_set in filters:
            filters[filters_set] = [max(int(f_ * filters_keep_percentage), min_filters) for f_ in filters[filters_set]]

    elif isinstance(filters, tuple):
        for filters_set in filters:
            new_filters_set = [max(int(f_ * filters_keep_percentage), min_filters) for f_ in filters_set]
            new_filters += (new_filters_set,)
        filters = new_filters

    latent_dim = max(int(latent_dim * filters_keep_percentage), min_filters)

    return filters, latent_dim


def variational_latent_block(latent_dim, x):
    """

    :param latent_dim:
    :param x:
    :return:
    """
    current_input_shape = x.shape[1:]  # last convolution shape
    num_of_units = tf.math.reduce_prod(current_input_shape)  # last convolution shape flattened

    encode_dense_block = Sequential([InputLayer(input_shape=current_input_shape)], name='flatten_and_dense_block')
    encode_dense_block.add(Flatten())
    encode_dense_block.add(Dense(latent_dim + latent_dim))  # latent mean and std
    x = encode_dense_block(x)

    latent_block = CustomLayers(name='variational_latent_block')(x)  # latent vector

    reshape_block = Sequential([InputLayer(input_shape=latent_block.shape[1:])], name='dense_and_reshape_block')
    reshape_block.add(Dense(units=num_of_units))
    reshape_block.add(LeakyReLU(alpha=0.2))
    reshape_block.add(Reshape(target_shape=current_input_shape))
    x = reshape_block(latent_block)

    return x


def check_assertions(model_name, apply_batch_norm, filters, mode):
    """

    :param model_name:
    :param apply_batch_norm:
    :param filters:
    :param mode:
    :return:
    """
    if model_name == 'functional_auto_encoder':
        if isinstance(filters, str):
            default_filters_value = 'functional_auto_encoder_default'
            variational_filters_value = 'functional_auto_encoder_variational'
            if 'advanced' in filters:
                raise ValueError(f"You are using 'functional_auto_encoder' and filters for 'advanced_auto_encoder'.")
    elif model_name == 'advanced_auto_encoder':
        if isinstance(filters, str):
            default_filters_value = 'advanced_auto_encoder_default'
            variational_filters_value = 'advanced_auto_encoder_variational'
            if 'functional' in filters:
                raise ValueError(f"You are using 'advanced_auto_encoder' and filters for 'functional_auto_encoder'.")
    else:
        raise ValueError(f"model_name '{model_name}' not implemented.")

    assert mode == 'ae' or mode == 'vae', f"Wrong value for argument 'mode' (value: {mode}). Should be one of 'ae'" \
                                          f"for a standard autoencoder, or 'vae' for variational autoencoder."
    if mode == 'ae':
        if isinstance(filters, str):
            assert 'variational' not in filters, \
                f"Wrong 'filters' for mode '{mode}' (value of 'filters': '{filters}'). Default filters for standard " \
                f"autoencoder have value '{default_filters_value}'."
    if mode == 'vae':
        if isinstance(filters, str):
            assert 'default' not in filters, \
                f"Wrong 'filters' for mode '{mode}' (value of 'filters': '{filters}'). Default filters for " \
                f"variational autoencoder have value '{variational_filters_value}'."
        assert not apply_batch_norm, "'apply_batch_norm' should be 'False' when using variational autoencoder."
