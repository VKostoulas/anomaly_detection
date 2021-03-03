"""
Contains all the settings stuff and stuff and even more stuff than this. Just kidding, it's just the settings man!
"""
MODE = 'experimental_train'

'------------------------------------------------Training parameters---------------------------------------------------'
EPOCHS = 30
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 128
OPTIMIZER = 'adam'
OPTIMIZER_PARAMS = {'lr': 0.001}
LOSS_FUNCTION = 'RelativeL1Loss'
LOSS_WEIGHTS = {}        # for mixed perceptual & L1 loss
SHUFFLE_SIZE = 5000
PROFILER_BATCHES_RANGE = [100, 110]
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
AUTOENCODER_PARAMS = {'filters_keep_percentage': 1.0, 'min_filters': 16, 'latent_dim': 512,
                      'kernel_initializer': 'glorot_uniform', 'filters': ([], [64, 128, 256, 512], [512, 256, 128, 64]),
                      'kernel_size': 2, 'mode': 'ae', 'apply_batch_norm': False}
CLASSIFIER_ARCHITECTURE = 'VGG19'
CLASSIFIER_PARAMS = {}
PERCEPTUAL_LOSS_LAYERS = ['block4_conv3']
'--------------------------------------------------Data parameters-----------------------------------------------------'
# INPUT_SHAPE = (96, 96, 3)
INPUT_SHAPE = (32, 32, 3)
# RESIZE_SHAPE = (64, 64)
# RESIZE_SHAPE = (128, 128)
RESIZE_SHAPE = None
# DATASET_NAME = 'malaria'
DATASET_NAME = 'cifar10'
# DATASET_NAME = 'patch_camelyon'
DATASET_SPLIT = 'validation'
# DATASET_SPLIT = 'train'
CLASSES = 10
TRAIN_PERCENTAGE = 0.5
'-----------------------------------------------------Data paths-------------------------------------------------------'
MAIN_PATH = 'C:/Users/Vagelis/PycharmProjects/anomaly_detection/'

# TRAIN_TFRECORDS_READ_PATH = 'C:/Users/Vagelis/tensorflow_datasets/patch_camelyon/2.0.0/*train.tfrecord*'
TRAIN_TFRECORDS_READ_PATH = MAIN_PATH + 'datasets/patch_camelyon/tfrecords/normal/*.tfrecord'
# TRAIN_TFRECORDS_READ_PATH = MAIN_PATH + 'malaria/tfrecords/uninfected/*.tfrecord'
VAL_TFRECORDS_READ_PATH = 'C:/Users/Vagelis/tensorflow_datasets/patch_camelyon/2.0.0/*validation.tfrecord*'
TEST_TFRECORDS_READ_PATH = 'C:/Users/Vagelis/tensorflow_datasets/patch_camelyon/2.0.0/*test.tfrecord*'

# TFRECORDS_SAVE_PATH = MAIN_PATH + DATASET_NAME + '/tfrecords/'
TFRECORDS_SAVE_PATH = MAIN_PATH + DATASET_NAME + '/tfrecords/'

PROJECT_MODEL_PATH = MAIN_PATH + 'trained_models/'
PROJECT_TENSORBOARD_PATH = MAIN_PATH + 'tensorboard/'
'----------------------------------------------------------------------------------------------------------------------'
