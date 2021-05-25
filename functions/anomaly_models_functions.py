"""
Functions to load architectures for both autoencoder and classifier.
"""
import os
from abc import ABC
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2DTranspose, Activation
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.nasnet import NASNetMobile, NASNetLarge
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from architectures.simple_autoencoder import simple_auto_encoder
from architectures.inception_autoencoder import inception_auto_encoder
from architectures.functional_autoencoder import functional_auto_encoder, functional_decode_step
from architectures.advanced_residual_auto_encoder import advanced_auto_encoder
from architectures.unet_autoencoder import unet_autoencoder
from functions.auto_encoder_functions import CustomLayers
from functions.common_functions import define_optimizer, define_loss_function, CustomInference2, extract_best_weights, \
    create_model_path


def define_image_size(s):
    if s.RESIZE_SHAPE:
        image_size = s.RESIZE_SHAPE + (s.INPUT_SHAPE[-1],)
    else:
        image_size = s.INPUT_SHAPE
    return image_size


def define_auto_encoder_architecture(architecture_name, image_size, auto_encoder_params):
    if architecture_name == 'FunctionalAutoEncoder':
        model = functional_auto_encoder(input_layer_shape=image_size, output_channels=image_size[-1],
                                        **auto_encoder_params)
    elif architecture_name == 'AdvancedAutoEncoder':
        model = advanced_auto_encoder(input_layer_shape=image_size, output_channels=image_size[-1],
                                      **auto_encoder_params)
    elif architecture_name == 'SimpleAutoEncoder':
        model = simple_auto_encoder(input_layer_shape=image_size, output_channels=image_size[-1],
                                    **auto_encoder_params)
    elif architecture_name == 'UnetAutoEncoder':
        model = unet_autoencoder(input_layer_shape=image_size, output_channels=image_size[-1],
                                 kernel_size=auto_encoder_params['kernel_size'])
    elif architecture_name == 'InceptionAutoEncoder':
        model = inception_auto_encoder(input_layer_shape=image_size, output_channels=image_size[-1],
                                       latent_dim=auto_encoder_params['latent_dim'])
    else:
        raise ValueError(f"Auto Encoder '{architecture_name}' is wrong or not implemented.")
    return model


def define_classifier_architecture(architecture_name, image_size, weights, classifier_kwargs=None):
    if architecture_name == 'MobileNet':
        model = MobileNetV2(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'VGG16':
        model = VGG16(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'VGG19':
        model = VGG19(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'NASNetMobile':
        model = NASNetMobile(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'NASNetLarge':
        model = NASNetLarge(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'InceptionV3':
        model = InceptionV3(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'InceptionResNetV2':
        model = InceptionResNetV2(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'Resnet50':
        model = ResNet50(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    elif architecture_name == 'EfficientNetB0':
        model = EfficientNetB0(input_shape=image_size, include_top=False, weights=weights, **classifier_kwargs)
    else:
        raise ValueError(f"Classifier '{architecture_name}' is wrong or not implemented.")

    return model


def build_autoencoder(architecture, image_size, ae_kwargs):
    autoencoder = define_auto_encoder_architecture(architecture, image_size, ae_kwargs)
    outputs = CustomLayers(name='concat_layer', dtype='float32')([autoencoder.input, autoencoder.output])
    return tf.keras.Model(inputs=autoencoder.inputs, outputs=outputs, name=autoencoder.name)


def build_classifier(architecture, loss_layers, image_size, weights='imagenet', clf_kwargs=None):
    classifier_model = define_classifier_architecture(architecture, image_size, weights, clf_kwargs)

    # Gather all selected layer outputs and combine them into one model output
    curr_layers_outs = []
    for intermediate_layer_name in loss_layers:
        curr_layers_outs.append(classifier_model.get_layer(intermediate_layer_name).output)

    classifier_model = tf.keras.Model(inputs=classifier_model.input, outputs=curr_layers_outs, name='classifier')

    classifier_input = tf.keras.layers.Input(shape=classifier_model.input.shape[1:])
    # rescale images-reconstructions to RGB values and preprocess them according to classifier preprocessing
    classifier_preprocessed_images = CustomLayers(name='rescale_and_preprocess_layer',
                                                  mode=architecture,
                                                  dtype='float32')(classifier_input)
    classifier_model = classifier_model(classifier_preprocessed_images, training=False)
    classifier_model = tf.keras.Model(classifier_input, classifier_model)

    # Weights should not be changed during training
    classifier_model.trainable = False

    return classifier_model


def build_anomaly_detection_model(s, auto_encoder, classifier, show_graph=False):
    anomaly_detector = PerceptualAutoEncoder(auto_encoder, classifier)

    if show_graph:
        auto_encoder.summary()
        classifier.summary()

    optimizer = define_optimizer(s.OPTIMIZER, s.OPTIMIZER_PARAMS)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    loss_function = define_loss_function(s.LOSS_FUNCTION)
    anomaly_detector.compile(optimizer=optimizer, loss=loss_function)

    return anomaly_detector


def inference_on_test_set(s, anomaly_detector, test_dataset, weights_path, verbose=1):
    if weights_path:
        # restore weights
        best_ae_weights, best_clf_weights, _ = extract_best_weights(weights_path, verbose=verbose)
        anomaly_detector.autoencoder.load_weights(best_ae_weights)
        anomaly_detector.classifier.load_weights(best_clf_weights)

    new_weights_path = create_model_path(s)
    cb = CustomInference2(val_data=test_dataset, save_model_path=new_weights_path, show_hist=s.SHOW_HISTOGRAMS_PLOTS)

    anomaly_detector.evaluate(test_dataset, callbacks=[cb])

    return new_weights_path


class PerceptualAutoEncoder(Model, ABC):
    def __init__(self, autoencoder, classifier):
        super(PerceptualAutoEncoder, self).__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

    def train_step(self, data):

        with tf.GradientTape() as tape:
            feature_maps = self.classifier(self.autoencoder(data[0], training=True), training=False)
            perceptual_loss = self.loss(y_true=None, y_pred=feature_maps)
            scaled_loss = self.optimizer.get_scaled_loss(perceptual_loss)

        scaled_grads = tape.gradient(scaled_loss, self.autoencoder.trainable_weights)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        return {"perceptual_loss": perceptual_loss}

    def test_step(self, data):
        feature_maps = self.classifier(self.autoencoder(data[0], training=False), training=False)
        perceptual_loss = self.loss(y_true=None, y_pred=feature_maps)

        return {'perceptual_loss': perceptual_loss}

    def call(self, inputs, training=None, mask=None):
        feature_maps = self.classifier(self.autoencoder(inputs, training=training), training=False)
        perceptual_loss = self.loss(y_true=None, y_pred=feature_maps)
        return perceptual_loss


class PerceptualEnsemble(Model, ABC):
    def __init__(self, *models):
        super(PerceptualEnsemble, self).__init__()
        self.models = models

    def test_step(self, data):
        all_scores = []
        output_dict = {}
        for i, model in enumerate(self.models):
            all_scores.append(model(data[0], training=False))
            output_dict[f'model_{i}_loss'] = all_scores[i]

        output_dict['perceptual_loss'] = tf.keras.layers.Average()(all_scores)

        return output_dict


class PerceptualMultiClassifier(Model, ABC):
    def __init__(self, autoencoder, classifier1, classifier2):
        super(PerceptualMultiClassifier, self).__init__()
        self.autoencoder = autoencoder
        self.classifier1 = classifier1
        self.classifier2 = classifier2

    def train_step(self, data):

        with tf.GradientTape() as tape:
            feature_maps1 = self.classifier1(self.autoencoder(data[0], training=True), training=False)
            perceptual_loss1 = self.loss(y_true=None, y_pred=feature_maps1)

            feature_maps2 = self.classifier2(self.autoencoder(data[0], training=True), training=False)
            perceptual_loss2 = self.loss(y_true=None, y_pred=feature_maps2)

            average_loss = (perceptual_loss1 + perceptual_loss2) / 2
            scaled_loss = self.optimizer.get_scaled_loss(average_loss)

        scaled_grads = tape.gradient(scaled_loss, self.autoencoder.trainable_weights)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        return {"perceptual_loss": average_loss, 'clf_1_loss': perceptual_loss1, 'clf_2_loss': perceptual_loss2}

    def test_step(self, data):
        feature_maps1 = self.classifier1(self.autoencoder(data[0], training=False), training=False)
        perceptual_loss1 = self.loss(y_true=None, y_pred=feature_maps1)

        feature_maps2 = self.classifier2(self.autoencoder(data[0], training=False), training=False)
        perceptual_loss2 = self.loss(y_true=None, y_pred=feature_maps2)

        average_loss = (perceptual_loss1 + perceptual_loss2) / 2

        return {'perceptual_loss': average_loss}


def modified_autoencoder(s):

    output_channels = define_image_size(s)[-1]
    decoder_filters = s.AUTOENCODER_PARAMS['filters'][2]
    kernel_size = s.AUTOENCODER_PARAMS['kernel_size']
    kernel_initializer = s.AUTOENCODER_PARAMS['kernel_initializer']

    conv_args = {'strides': 2, 'kernel_size': kernel_size, 'padding': 'same', 'kernel_initializer': kernel_initializer}

    encoder = build_classifier(s, weights=None)
    encoder.trainable = True
    x = encoder.output

    for i, _f in enumerate(decoder_filters):
        x = functional_decode_step(input_volume=x, filters=_f, apply_batch_norm=False, conv_args=conv_args,
                                   name='up{}'.format(i))

    # Last layer of Decoder
    model_output = Conv2DTranspose(filters=output_channels, use_bias=True, name='final_conv', **conv_args)(x)
    model_output = Activation('tanh', dtype='float32', name='tanh')(model_output)
    outputs = CustomLayers(name='concat_layer', dtype='float32')([encoder.input, model_output])

    return tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name='auto_encoder')

