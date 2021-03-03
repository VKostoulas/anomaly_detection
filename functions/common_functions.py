import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import optimizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from functions.inference_functions import model_infer_loop, inference_results


def create_model_path(s):
    if s.SAVE_MODELS:
        model_save_time = datetime.today().strftime('%Y-%m-%d') + '_' + datetime.now().time().strftime('%H-%M-%S')
        model_path = s.PROJECT_MODEL_PATH + model_save_time
    else:
        model_path = None

    return model_path


def define_optimizer(opt_name, opt_kwargs):

    if opt_name == 'sgd':
        optimizer = optimizers.SGD(**opt_kwargs)
    elif opt_name == 'adam':
        optimizer = optimizers.Adam(**opt_kwargs)
    elif opt_name == 'adagrad':
        optimizer = optimizers.Adagrad(**opt_kwargs)
    elif opt_name == 'rmsprop':
        optimizer = optimizers.RMSprop(**opt_kwargs)
    elif opt_name == 'adadelta':
        optimizer = optimizers.Adadelta(**opt_kwargs)
    elif opt_name == 'nadam':
        optimizer = optimizers.Nadam(**opt_kwargs)
    elif opt_name == 'adamax':
        optimizer = optimizers.Adamax(**opt_kwargs)
    else:
        raise ValueError(f"Optimizer {opt_name} is wrong or not implemented.")
    return optimizer


def relative_l1_loss(y_true, y_pred):

    del y_true
    y_pred = tf.cast(y_pred, tf.float32) if y_pred.dtype == tf.float16 else y_pred  # for mixed precision
    y_pred = tf.keras.layers.Flatten(dtype='float32')(y_pred) if len(y_pred.shape) > 2 else y_pred
    y_pred = tf.split(y_pred, num_or_size_splits=2, axis=0)  # split to image and reconstruction features

    return tf.divide(tf.norm(tf.subtract(y_pred[0], y_pred[1]), ord=1, axis=-1),
                     tf.norm(y_pred[0], ord=1, axis=-1))


def custom_rmse(y_true, y_pred):
    del y_true
    y_pred_shape = tf.cast(y_pred.shape[1:], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) if y_pred.dtype == tf.float16 else y_pred  # for mixed precision
    y_pred = tf.keras.layers.Flatten(dtype='float32')(y_pred) if len(y_pred.shape) > 2 else y_pred
    y_pred = tf.split(y_pred, num_or_size_splits=2, axis=0)  # split to image and reconstruction features
    return tf.sqrt(tf.divide(tf.square(tf.norm(tf.subtract(y_pred[0], y_pred[1]), ord=2, axis=-1)), tf.reduce_prod(y_pred_shape)))


def relative_l2_loss(y_true, y_pred):
    del y_true
    y_pred = tf.cast(y_pred, tf.float32) if y_pred.dtype == tf.float16 else y_pred  # for mixed precision
    y_pred = tf.keras.layers.Flatten(dtype='float32')(y_pred) if len(y_pred.shape) > 2 else y_pred
    y_pred = tf.split(y_pred, num_or_size_splits=2, axis=0)  # split to image and reconstruction features

    return tf.divide(tf.norm(tf.subtract(y_pred[0], y_pred[1]), ord=2, axis=-1),
                     tf.norm(y_pred[0], ord=2, axis=-1))


def define_loss_function(loss_name):
    if loss_name == 'RMSE':
        loss_function = custom_rmse
    elif loss_name == 'RelativeL2Loss':
        loss_function = relative_l2_loss
    elif loss_name == 'RelativeL1Loss':
        loss_function = relative_l1_loss
    elif loss_name == 'PerceptualAndL1':
        loss_function = {'perceptual': relative_l1_loss, 'relative_l1': relative_l1_loss}
    else:
        raise ValueError(f"Loss function {loss_name} is wrong or not implemented.")

    return loss_function


def define_mixed_precision_policy(use_mixed_precision):

    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)


def define_callbacks(s, val_dataset, auto_encoder, model_path, verbose=1):

    callbacks = []

    if s.USE_LR_DECAY:

        def lr_time_based_decay(epoch, learning_rate, decay=0.01):
            return learning_rate * (1 / (1 + decay * epoch))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=verbose))

    if s.USE_TENSORBOARD:
        tensorboard_cb = define_tensorboard_callback(s)
        callbacks.append(tensorboard_cb)

    if s.SHOW_RECONSTRUCTIONS:
        callbacks.append(DisplayCallback(data=val_dataset, infer_model=auto_encoder,
                                         num_of_samples=2, sample_indexes=None))

    cb2 = CustomInference2(val_data=val_dataset, verbose=verbose,
                           save_model_path=model_path, show_hist=s.SHOW_HISTOGRAMS_PLOTS)
    callbacks.append(cb2)

    return callbacks


def define_tensorboard_callback(s):

    tensorboard_log_dir = os.path.join(s.PROJECT_TENSORBOARD_PATH, 'keras_fit')
    if s.PROFILER_BATCHES_RANGE[0] > s.PROFILER_BATCHES_RANGE[1]:
        raise ValueError('PROFILER_BATCHES_RANGE is not a valid range %s' % s.PROFILER_BATCHES_RANGE)
    if s.PROFILER_BATCHES_RANGE[1] == 0:
        profile_batches = 0  # disable profiler with [0, 0]
    else:
        profile_batches = str(s.PROFILER_BATCHES_RANGE[0]) + ',' + str(s.PROFILER_BATCHES_RANGE[1])
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,
                                                    histogram_freq=1,
                                                    profile_batch=profile_batches,
                                                    write_graph=True)
    return tensorboard_cb


def calculate_class_percentage(k_folds, k_step):
    if k_step + 100 - 100 / k_folds > 100:
        class_percentage1 = str(k_step) + ':' + str(100)
        class_percentage2 = ':' + str(100 - 100 / k_folds - (100 - k_step))
        train_class_percentage = class_percentage1 + ',' + class_percentage2
    else:
        train_class_percentage = str(k_step) + ':' + str(k_step + 100 - 100 / k_folds)

    if k_step > 0:
        val_class_percentage = str(k_step - 100 / k_folds) + ':' + str(k_step)
    else:
        val_class_percentage = str(k_step + 100 - 100 / k_folds) + ':' + str(k_step + 100)

    return train_class_percentage, val_class_percentage


class CustomInference(tf.keras.callbacks.Callback):
    """
    At the end of every epoch, infer on 1 or more validation sets.
    """
    def __init__(self, s, data, loss_func):
        super().__init__()
        self.s = s
        self.data = data
        self.loss_func = loss_func

    def on_epoch_end(self, epoch, logs=None):   # pylint: disable=unused-argument
        """
        At the end of every epoch, infer on every validation set.

        :param epoch:
        :param logs:
        :return:
        """
        model_infer_loop(self.model, self.data, self.loss_func, 'Epoch {} on Validation set'.format(epoch),
                         self.s.NUM_SAMPLES_TO_PLOT, self.s.SHOW_HISTOGRAMS_PLOTS)


class CustomInference2(tf.keras.callbacks.Callback):
    def __init__(self, val_data, verbose=1, save_model_path=None, show_hist=False):
        """

        :param val_data:
        """
        super().__init__()
        self.data = val_data
        self.verbose = verbose
        self.save_model_path = save_model_path
        self.show_hist = show_hist
        self.anomaly_scores = []
        self.labels = []
        self.curr_auc_value = 0

    def on_test_batch_end(self, batch, logs=None):
        self.anomaly_scores.extend(logs['perceptual_loss'])

    def on_test_end(self, logs=None):

        for batch in self.data:
            labels = batch[1]
            labels = [1 if label[1] == 1 else 0 for label in labels]
            self.labels.extend(labels)

        curr_auc_value = inference_results(self.anomaly_scores, self.labels, 'Anomaly Scores', show_hist=self.show_hist,
                                           verbose=self.verbose)

        if self.save_model_path:
            # save model with best auc
            if curr_auc_value >= self.curr_auc_value:
                if self.verbose > 0:
                    print(f'Saving weights to {self.save_model_path}')
                self.curr_auc_value = curr_auc_value
                self.model.autoencoder.save_weights(self.save_model_path + f'/autoencoder_auc_{curr_auc_value:.4f}')
                if hasattr(self.model, 'classifier'):
                    self.model.classifier.save_weights(self.save_model_path + f'/classifier_auc_{curr_auc_value:.4f}')
                elif hasattr(self.model, 'classifier1') and hasattr(self.model, 'classifier2'):
                    self.model.classifier1.save_weights(self.save_model_path + f'/classifier1_auc_{curr_auc_value:.4f}')
                    self.model.classifier2.save_weights(self.save_model_path + f'/classifier2_auc_{curr_auc_value:.4f}')

        self.anomaly_scores = []
        self.labels = []

        del labels, curr_auc_value


class DisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self, data, infer_model, num_of_samples, sample_indexes):
        """

        :param data:
        :param num_of_samples:
        :param sample_index:
        """
        super().__init__()
        self.data = data
        self.infer_model = infer_model
        self.num_of_samples = num_of_samples
        self.sample_indexes = sample_indexes

    def on_train_begin(self, logs=None):    # pylint: disable=unused-argument
        """

        :param logs:
        :return:
        """
        self.plot_images(epoch=0)

    def on_epoch_end(self, epoch, logs=None):    # pylint: disable=unused-argument
        """

        :param epoch:
        :param logs:
        :return:
        """
        self.plot_images(epoch)

    def plot_images(self, epoch):
        """

        :param epoch:
        :return:
        """
        for batch in self.data:
            # define images and their labels
            sample_indexes = self.__define_sample_indexes(batch)
            if None not in sample_indexes:
                images = list(batch[0][sample_index] for sample_index in sample_indexes)
                labels = list(batch[1][sample_index] for sample_index in sample_indexes)
                break
        labels = [1 if label[1] == 1 else 0 for label in labels]

        _f, _ax = plt.subplots(self.num_of_samples, 2)
        _f.suptitle(f'Epoch {epoch}')

        for i in range(self.num_of_samples):
            sample_image, label = images[i], labels[i]
            sample_image = sample_image[np.newaxis, :]
            img_rcn = self.infer_model(sample_image, training=False).numpy().astype('float32')
            img_rcn = tf.split(img_rcn, num_or_size_splits=2, axis=0)
            image, reconstruction = np.squeeze(img_rcn[0], axis=0), np.squeeze(img_rcn[1], axis=0)

            if self.num_of_samples == 1:
                _ax[i].imshow(np.clip(image, a_min=0, a_max=1))
                _ax[i].set_title(f'Image (label: {label})')
                _ax[i].axis('off')
                _ax[i+1].imshow(np.clip(reconstruction, a_min=0, a_max=1))
                _ax[i+1].set_title(f'Reconstruction (label: {label})')
                _ax[i+1].axis('off')
            else:
                _ax[i, 0].imshow(np.clip(image, a_min=0, a_max=1))
                _ax[i, 0].set_title(f'Image (label: {label})')
                _ax[i, 0].axis('off')
                _ax[i, 1].imshow(np.clip(reconstruction, a_min=0, a_max=1))
                _ax[i, 1].set_title(f'Reconstruction (label: {label})')
                _ax[i, 1].axis('off')
        plt.show()
        plt.close()

    def __define_sample_indexes(self, batch):

        if self.sample_indexes:
            sample_indexes = self.sample_indexes
        else:
            # choose one image for every class
            sample_indexes = {'normal': None, 'tumor': None}
            for i, label in enumerate(batch[1]):
                label = [1 if label[1] == 1.0 else 0][0]
                if label == 0:
                    if sample_indexes['normal'] is None:
                        sample_indexes['normal'] = i
                else:
                    if sample_indexes['tumor'] is None:
                        sample_indexes['tumor'] = i

                if sample_indexes['normal'] and sample_indexes['tumor']:
                    break

            sample_indexes = [sample_indexes['normal'], sample_indexes['tumor']]

        return sample_indexes


def extract_best_weights(weights_folder, verbose=1):
    # load model with best auc score
    folder_files = os.listdir(weights_folder)
    best_ae_auc_score = 0
    best_clf_auc_score = 0
    best_ae_weights_file = ''
    best_clf_weights_file = ''

    for file in folder_files:
        if file.endswith('.index'):
            if 'autoencoder' in file:
                curr_auc = float(re.findall(r"[+-]?\d+\.\d+", file)[0])
                if curr_auc >= best_ae_auc_score:
                    best_ae_auc_score = curr_auc
                    best_ae_weights_file = file

            if 'classifier' in file:
                curr_auc = float(re.findall(r"[+-]?\d+\.\d+", file)[0])
                if curr_auc >= best_clf_auc_score:
                    best_clf_auc_score = curr_auc
                    best_clf_weights_file = file
    assert best_ae_auc_score == best_clf_auc_score, \
        'Weights file of classifier should be the corresponding of autoencoder weights file'

    best_ae_weights_path = weights_folder + '/' + best_ae_weights_file[:-6]
    if verbose > 0:
        print('\nbest autoencoder weights: ', best_ae_weights_path)

    best_clf_weights_path = weights_folder + '/' + best_clf_weights_file[:-6]
    if 'classifier1' in best_clf_weights_path or 'classifier2' in best_clf_weights_path:
        best_clf1_weights_path = best_clf_weights_path if 'classifier1' in best_clf_weights_path else \
            best_clf_weights_path.replace('classifier2', 'classifier1')
        best_clf2_weights_path = best_clf1_weights_path.replace('classifier1', 'classifier2')
        outputs = best_ae_weights_path, best_clf1_weights_path, best_clf2_weights_path, best_ae_auc_score
        if verbose > 0:
            print('best classifier1 weights: ', best_clf1_weights_path)
            print('best classifier2 weights: ', best_clf2_weights_path)
    else:
        outputs = best_ae_weights_path, best_clf_weights_path, best_ae_auc_score
        if verbose > 0:
            print('best classifier weights: ', best_clf_weights_path)

    return outputs


