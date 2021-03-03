import os
import numpy as np
from abc import ABC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)\
        (tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)\
        (tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) \
        (tf.ones_like(fake_output), fake_output)


class GAN(Model, ABC):
    def __init__(self, s, autoencoder, classifier):
        super(GAN, self).__init__()
        self.s = s
        self.autoencoder = autoencoder
        self.classifier = classifier

    def compile(self, ae_optimizer, clf_optimizer, ae_loss_fn, clf_loss_fn):
        super(GAN, self).compile()
        self.ae_optimizer = ae_optimizer
        self.clf_optimizer = clf_optimizer
        self.ae_loss_fn = ae_loss_fn
        self.clf_loss_fn = clf_loss_fn

    def train_step(self, data):

        real_images = data[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            real_and_fake_images = self.autoencoder(real_images, training=True)
            real_and_fake_outputs = self.classifier(real_and_fake_images, training=True)
            predictions_split = tf.split(real_and_fake_outputs[1], num_or_size_splits=2, axis=0)

            ae_loss = self.ae_loss_fn(None, real_and_fake_outputs[0])
            ae_loss += 0.1 * self.clf_loss_fn(tf.ones_like(predictions_split[1]), predictions_split[1])

            clf_real_loss = self.clf_loss_fn(tf.ones_like(predictions_split[0]), predictions_split[0])
            clf_fake_loss = self.clf_loss_fn(tf.zeros_like(predictions_split[1]), predictions_split[1])
            clf_loss = clf_real_loss + clf_fake_loss

        gradients_of_autoencoder = gen_tape.gradient(ae_loss, self.autoencoder.trainable_variables)
        gradients_of_classifier = disc_tape.gradient(clf_loss, self.classifier.trainable_variables)

        self.ae_optimizer.apply_gradients(zip(gradients_of_autoencoder, self.autoencoder.trainable_variables))
        self.clf_optimizer.apply_gradients(zip(gradients_of_classifier, self.classifier.trainable_variables))
        return {"auto_encoder_loss": ae_loss, "classifier_loss": clf_loss}

    def test_step(self, data):

        real_images = data[0]

        real_and_fake_images = self.autoencoder(real_images, training=False)
        real_and_fake_outputs = self.classifier(real_and_fake_images, training=False)
        predictions_split = tf.split(real_and_fake_outputs[1], num_or_size_splits=2, axis=0)

        anomaly_scores = self.ae_loss_fn(None, real_and_fake_outputs[0])
        anomaly_scores += 0.1 * self.clf_loss_fn(tf.ones_like(predictions_split[1]), predictions_split[1])

        return {'perceptual_loss': anomaly_scores}


class PerceptualGAN(Model, ABC):
    def __init__(self, s, autoencoder, classifier):
        super(PerceptualGAN, self).__init__()
        self.s = s
        self.autoencoder = autoencoder
        self.classifier = classifier

    def compile(self, optimizer, ae_loss_fn, clf_loss_fn):
        super(PerceptualGAN, self).compile()
        self.optimizer = optimizer
        self.ae_loss_fn = ae_loss_fn
        self.clf_loss_fn = clf_loss_fn

    def train_step(self, data):

        real_images = data[0]
        batch_size = tf.shape(real_images)[0]

        # Assemble labels discriminating actual from generated images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels = tf.squeeze(labels)

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the classifier
        with tf.GradientTape() as tape:
            predictions = self.classifier(self.autoencoder(real_images, training=False), training=True)[1]
            imgs_and_rcns_predictions = tf.split(predictions, num_or_size_splits=2, axis=0)
            real_loss = self.clf_loss_fn(tf.ones_like(imgs_and_rcns_predictions[0]), imgs_and_rcns_predictions[0])
            fake_loss = self.clf_loss_fn(tf.zeros_like(imgs_and_rcns_predictions[1]), imgs_and_rcns_predictions[1])
            cl_loss = real_loss + fake_loss
            # cl_scaled_loss = self.cl_optimizer.get_scaled_loss(cl_loss)
        grads = tape.gradient(cl_loss, self.classifier.trainable_weights)
        # grads = self.cl_optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.classifier.trainable_weights))

        misleading_labels = tf.ones((batch_size, 1))
        misleading_labels = tf.squeeze(misleading_labels)

        # Train the auto encoder
        with tf.GradientTape() as tape:
            predictions = self.classifier(self.autoencoder(real_images, training=True), training=False)
            ae_loss = self.ae_loss_fn(y_true=None, y_pred=predictions[0])
            imgs_and_rcns_predictions = tf.split(predictions[1], num_or_size_splits=2, axis=0)
            # ae_loss = ae_loss + self.cl_loss_fn(y_true=misleading_labels, y_pred=imgs_and_rcns_predictions[0])
            ae_loss = ae_loss + 0.01 * self.clf_loss_fn(y_true=misleading_labels, y_pred=imgs_and_rcns_predictions[1])
            # ae_scaled_loss = self.ae_optimizer.get_scaled_loss(ae_loss)
        grads = tape.gradient(ae_loss, self.autoencoder.trainable_weights)
        # grads = self.ae_optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        return {"auto_encoder_loss": ae_loss, "classifier_loss": cl_loss}

    def test_step(self, data):

        real_images = data[0]
        batch_size = tf.shape(real_images)[0]
        misleading_labels = tf.ones((batch_size, 1))
        misleading_labels = tf.squeeze(misleading_labels)

        predictions = self.classifier(self.autoencoder(real_images, training=False), training=False)
        anomaly_scores = self.ae_loss_fn(y_true=None, y_pred=predictions[0])
        imgs_and_rcns_predictions = tf.split(predictions[1], num_or_size_splits=2, axis=0)
        # anomaly_scores += self.cl_loss_fn(misleading_labels, imgs_and_rcns_predictions[0])
        anomaly_scores += self.clf_loss_fn(misleading_labels, imgs_and_rcns_predictions[1])

        return {'perceptual_loss': anomaly_scores}


class CustomLRScheduler(tf.keras.callbacks.Callback):

    def __init__(self, schedule, optimizer_mode, verbose=0):
        super(CustomLRScheduler, self).__init__()
        self.schedule = schedule
        self.optimizer_mode = optimizer_mode
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, self.optimizer_mode):
            raise ValueError(f'Model must have a "{self.optimizer_mode}" attribute.')
        optimizer_object = getattr(self.model, self.optimizer_mode)
        if not hasattr(optimizer_object, 'lr'):
            raise ValueError(f'{self.optimizer_mode} must have a "lr" attribute.')
        try:  # new API
            lr = float(K.get_value(optimizer_object.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(getattr(getattr(self.model, self.optimizer_mode), 'lr'), K.get_value(lr))
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1:05d}: reducing {self.optimizer_mode} learning rate to {lr}.')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(getattr(getattr(self.model, self.optimizer_mode), 'lr'))
