# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Press Ctrl+F8 to toggle a breakpoint.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import os
import numpy as np
from python_settings import settings as s
import my_settings as local_settings

from functions.dataset_functions import create_tfrecords_from_tfdatasets
from functions.training_functions import train_anomaly_detection_model, k_fold_training, gans_training, \
    experimental_training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

if __name__ == '__main__':
    s.configure(local_settings)

    if s.MODE == 'test':
        print('running tests')
        # classes_dict = {'normal': 0, 'tumor': 1}
        classes_dict = {'parasitized': 0, 'uninfected': 1}
        create_tfrecords_from_tfdatasets(s.DATASET_NAME, s.DATASET_SPLIT, classes_dict, s.TFRECORDS_SAVE_PATH)

    elif s.MODE == 'simple_train':

        # Example
        experiment_16_global_params = {'TRAIN_BATCH_SIZE': 4,
                                       'INFER_BATCH_SIZE': 32,
                                       'OPTIMIZER_PARAMS': {'lr': 0.0001},
                                       'AUTOENCODER_ARCHITECTURE': 'AdvancedAutoEncoder',
                                       'AUTOENCODER_PARAMS':
                                           {'filters_keep_percentage': 1.0, 'min_filters': 16, 'latent_dim': 512,
                                            'kernel_initializer': 'glorot_uniform',
                                            'filters': ([64], [64, 128, 128, 256, 256], [256, 256, 128, 128, 64], []),
                                            'kernel_size': 3, 'mode': 'vae', 'apply_batch_norm': False},
                                       'CLASSIFIER_ARCHITECTURE': 'MobileNet',
                                       'CLASSIFIER_PARAMS': {},
                                       'PERCEPTUAL_LOSS_LAYERS': ['block_7_depthwise_relu'],
                                       'RESIZE_SHAPE': (128, 128),
                                       'TRAIN_PERCENTAGE': 0.3
                                       }

        experiments_params = [experiment_16_global_params]

        for params in experiments_params:
            for arg in params:
                setattr(s, arg, params[arg])
            try:
                train_anomaly_detection_model(s)
            except tf.python.framework.errors_impl.ResourceExhaustedError:
                print('\n program stopped because of memory error!')

    elif s.MODE == 'k_fold_train':

        k_fold_training(s)

    elif s.MODE == 'gans_train':
        gans_training(s)

    elif s.MODE == 'experimental_train':

        # mode = 'different_losses'
        mode = 'perceptual_ensemble'

        if mode == 'perceptual_ensemble':

            # Example
            # global settings
            image_size = (128, 128, 3)
            global_params = {'TRAIN_BATCH_SIZE': 64, 'INFER_BATCH_SIZE': 128, 'OPTIMIZER_PARAMS': {'lr': 0.001},
                             'INPUT_SHAPE': (96, 96, 3), 'RESIZE_SHAPE': image_size[:-1]}
            # dataset arguments
            # data_params = {'s': s, 'train_class_percentage': ':90', 'val_class_percentage': '90:',
            #                'normal_class': 0, 'anomaly_classes': [5]}
            # autoencoder arguments
            ae1_params = {'architecture': 'SimpleAutoEncoder',
                          'image_size': image_size,
                          'ae_kwargs': {'filters_keep_percentage': 1.0, 'min_filters': 16, 'latent_dim': 512,
                                        'kernel_initializer': 'glorot_uniform',
                                        'filters': ([], [64, 128, 128, 256], [256, 128, 128, 64]),
                                        'kernel_size': 2, 'mode': 'ae', 'apply_batch_norm': False}}
            # classifier arguments
            # clf1_params = {'architecture': 'VGG19', 'loss_layers': ['block4_conv3'],
            #                'image_size': image_size, 'clf_kwargs': s.CLASSIFIER_PARAMS}
            clf2_params = {'architecture': 'MobileNet', 'loss_layers': ['block_7_depthwise_relu'],
                           'image_size': image_size, 'clf_kwargs': s.CLASSIFIER_PARAMS}
            clf3_params = {'architecture': 'MobileNet', 'loss_layers': ['block_7_depthwise_relu'],
                           'image_size': image_size, 'clf_kwargs': s.CLASSIFIER_PARAMS}
            clf4_params = {'architecture': 'MobileNet', 'loss_layers': ['block_7_depthwise_relu'],
                           'image_size': image_size, 'clf_kwargs': s.CLASSIFIER_PARAMS}
            clf5_params = {'architecture': 'MobileNet', 'loss_layers': ['block_7_depthwise_relu'],
                           'image_size': image_size, 'clf_kwargs': s.CLASSIFIER_PARAMS}
            ae_params = [ae1_params, ae1_params]
            clf_params = [clf2_params, clf3_params, clf4_params, clf5_params]

            data_params = {'s': s}
            experimental_training(s, mode=mode, data_params=data_params, ae_params=ae_params,
                                  clf_params=clf_params, global_params=global_params)

        elif mode == 'different_losses':

            # Example
            global1_params = {'LOSS_FUNCTION': 'RelativeL1Loss'}
            global2_params = {'LOSS_FUNCTION': 'RelativeL2Loss'}
            global3_params = {'LOSS_FUNCTION': 'RMSE'}
            global4_params = {'LOSS_FUNCTION': 'RelativeL1Loss',
                              'AUTOENCODER_PARAMS': {'filters_keep_percentage': 1.0, 'min_filters': 16,
                                                     'latent_dim': 512,
                                                     'kernel_initializer': 'glorot_uniform',
                                                     'filters': ([],
                                                                 [64, 128, 128, 256, 512], [512, 256, 128, 128, 64]),
                                                     'kernel_size': 2, 'mode': 'ae', 'apply_batch_norm': False},
                              'RESIZE_SHAPE': (128, 128)
                              }
            global5_params = {'LOSS_FUNCTION': 'RelativeL2Loss',
                              'AUTOENCODER_PARAMS': {'filters_keep_percentage': 1.0, 'min_filters': 16,
                                                     'latent_dim': 512,
                                                     'kernel_initializer': 'glorot_uniform',
                                                     'filters': ([],
                                                                 [64, 128, 128, 256, 512], [512, 256, 128, 128, 64]),
                                                     'kernel_size': 2, 'mode': 'ae', 'apply_batch_norm': False},
                              'RESIZE_SHAPE': (128, 128)
                              }
            global6_params = {'LOSS_FUNCTION': 'RMSE',
                              'AUTOENCODER_PARAMS': {'filters_keep_percentage': 1.0, 'min_filters': 16,
                                                     'latent_dim': 512,
                                                     'kernel_initializer': 'glorot_uniform',
                                                     'filters': ([],
                                                                 [64, 128, 128, 256, 512], [512, 256, 128, 128, 64]),
                                                     'kernel_size': 2, 'mode': 'ae', 'apply_batch_norm': False},
                              'RESIZE_SHAPE': (128, 128)
                              }
            global_params = [global1_params, global2_params, global3_params,
                             global4_params, global5_params, global6_params]

            data_params = {'s': s}
            for params in global_params:
                experimental_training(s, mode=mode, data_params=data_params, global_params=params)