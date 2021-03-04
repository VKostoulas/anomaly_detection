"""
This file contains all global settings that are used in the project.
"""
MODE = 'simple_train'

'------------------------------------------------Training parameters---------------------------------------------------'
EPOCHS = 20
TRAIN_BATCH_SIZE = 32
INFER_BATCH_SIZE = 64
OPTIMIZER = 'adam'
OPTIMIZER_PARAMS = {'lr': 0.0005}
LOSS_FUNCTION = 'RelativeL1Loss'
SHUFFLE_SIZE = 10000
PROFILER_BATCHES_RANGE = [150, 160]
K_FOLD_NUM = 3

USE_MIXED_PRECISION = False
USE_AUGMENTATION = False
USE_LR_DECAY = False
USE_TENSORBOARD = False
SAVE_MODELS = True
SHOW_HISTOGRAMS_PLOTS = False
SHOW_RECONSTRUCTIONS = False
NUM_SAMPLES_TO_PLOT = None
'-------------------------------------------------Model parameters-----------------------------------------------------'
AUTOENCODER_ARCHITECTURE = 'SimpleAutoEncoder'
AUTOENCODER_PARAMS = {'filters_keep_percentage': 1.0, 'min_filters': 16, 'latent_dim': 128,
                      'kernel_initializer': 'glorot_uniform', 'filters': ([], [64, 64, 64, 64], [64, 64, 64, 64]),
                      'kernel_size': 2, 'mode': 'ae', 'apply_batch_norm': False}
CLASSIFIER_ARCHITECTURE = 'VGG19'
CLASSIFIER_PARAMS = {}
PERCEPTUAL_LOSS_LAYERS = ['block3_conv2']
'--------------------------------------------------Data parameters-----------------------------------------------------'
INPUT_SHAPE = (96, 96, 3)
RESIZE_SHAPE = None
DATASET_NAME = 'patch_camelyon'
DATASET_SPLIT = ''
CLASSES = 2
TRAIN_PERCENTAGE = 1.0
'-----------------------------------------------------Data paths-------------------------------------------------------'
MAIN_PATH = ''

TRAIN_TFRECORDS_READ_PATH = MAIN_PATH + './custom_datasets/patch_camelyon/tfrecords/normal/*.tfrecord'
VAL_TFRECORDS_READ_PATH = './tensorflow_datasets/patch_camelyon/2.0.0/*validation.tfrecord*'
TEST_TFRECORDS_READ_PATH = './tensorflow_datasets/patch_camelyon/2.0.0/*test.tfrecord*'

TFRECORDS_SAVE_PATH = MAIN_PATH + DATASET_NAME + '/tfrecords/'

PROJECT_MODEL_PATH = MAIN_PATH + 'trained_models/'
PROJECT_TENSORBOARD_PATH = MAIN_PATH + 'tensorboard/'
'----------------------------------------------------------------------------------------------------------------------'
