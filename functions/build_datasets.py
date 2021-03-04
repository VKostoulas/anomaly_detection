import os
import importlib
from copy import deepcopy

from functions.dataset_functions import augmentation_func, preprocess_img_label, choose_tfrecords, \
    _parse_image_function, select_examples_from_label, extract_percentage_of_data, convert_labels_to_zeros_and_ones

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds


def choose_dataset(s, train_class_percentage=None, val_class_percentage=None, normal_class=None,
                   anomaly_classes=None):

    if s.DATASET_NAME in ['cifar10', 'mnist']:

        train_dataset = build_keras_dataset(dataset_name=s.DATASET_NAME, dataset_part='training',
                                            class_percentage=train_class_percentage, normal_class_label=normal_class,
                                            anomaly_class_labels=[], batch_size=s.TRAIN_BATCH_SIZE,
                                            num_of_classes=s.CLASSES, use_augmentation=s.USE_AUGMENTATION,
                                            images_shuffle_size=s.SHUFFLE_SIZE, resize_shape=s.RESIZE_SHAPE)

        val_dataset = build_keras_dataset(dataset_name=s.DATASET_NAME, dataset_part='training',
                                          class_percentage=val_class_percentage, normal_class_label=normal_class,
                                          anomaly_class_labels=anomaly_classes, batch_size=s.INFER_BATCH_SIZE,
                                          num_of_classes=s.CLASSES, use_augmentation=False,
                                          images_shuffle_size=None, resize_shape=s.RESIZE_SHAPE)

        test_dataset = build_keras_dataset(dataset_name=s.DATASET_NAME, dataset_part='test',
                                           class_percentage=':', normal_class_label=normal_class,
                                           anomaly_class_labels=-1, batch_size=s.INFER_BATCH_SIZE,
                                           num_of_classes=s.CLASSES, use_augmentation=False,
                                           images_shuffle_size=None, resize_shape=s.RESIZE_SHAPE)

    elif s.DATASET_NAME in ['patch_camelyon']:

        # Create datasets
        train_dataset, _ = build_custom_dataset(tfrecords_paths=s.TRAIN_TFRECORDS_READ_PATH,
                                                percentage=s.TRAIN_PERCENTAGE, set_name='Train',
                                                use_augmentation=s.USE_AUGMENTATION, num_of_classes=s.CLASSES,
                                                batch_size=s.TRAIN_BATCH_SIZE, images_per_tfrecord=5000,
                                                images_shuffle_size=s.SHUFFLE_SIZE, shuffle_tfrecords=True,
                                                return_metadata=False, resize_shape=s.RESIZE_SHAPE)
        # for some reason, using resize in training but not in validation seems to improve AUC a lot
        val_dataset, metadata, val_num_of_batches = build_tf_dataset(dataset_name=s.DATASET_NAME,
                                                                     dataset_split='validation'+s.DATASET_SPLIT,
                                                                     batch_size=s.INFER_BATCH_SIZE,
                                                                     num_of_classes=s.CLASSES, use_augmentation=False,
                                                                     resize_shape=s.RESIZE_SHAPE)

        test_dataset, _, test_num_of_batches = build_tf_dataset(dataset_name=s.DATASET_NAME,
                                                                dataset_split='test',
                                                                batch_size=s.INFER_BATCH_SIZE,
                                                                num_of_classes=s.CLASSES, use_augmentation=False,
                                                                resize_shape=s.RESIZE_SHAPE)
        print('Number of classes: {}'.format(metadata.features['label'].num_classes))

    else:
        raise ValueError(f'Dataset name {s.DATASET_NAME} is wrong or not implemented.')

    return train_dataset, val_dataset, test_dataset


def build_keras_dataset(dataset_name, dataset_part, class_percentage, normal_class_label, anomaly_class_labels,
                        batch_size, num_of_classes, use_augmentation, images_shuffle_size=None, resize_shape=None):

    dataset = importlib.import_module('tensorflow.keras.datasets.' + dataset_name).load_data()

    if dataset_part == 'training':
        (images, labels), _ = dataset
    elif dataset_part == 'test':
        _, (images, labels) = dataset
    else:
        raise ValueError('Wrong mode!')

    preprocess_func_args = {'resize_shape': resize_shape, 'num_of_classes': num_of_classes}
    anomaly_class_labels = [i for i in list(range(10)) if i != normal_class_label] if anomaly_class_labels == -1 \
        else anomaly_class_labels

    images, labels = select_examples_from_label(normal_class_label, anomaly_class_labels, images, labels)
    images, labels = extract_percentage_of_data(images, labels, class_percentage)
    labels = convert_labels_to_zeros_and_ones(normal_class_label, labels)

    classes_in_dataset = str(normal_class_label)
    for class_label in anomaly_class_labels:
        classes_in_dataset += ', ' + str(class_label)
    print(f"{dataset_name} {dataset_part} set (classes: {classes_in_dataset}) has {images.shape[0]} examples")

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(images_shuffle_size) if images_shuffle_size else dataset
    augm_func = augmentation_func if use_augmentation else preprocess_img_label
    dataset = dataset.map(lambda x, y: (x, tf.squeeze(y)), num_parallel_calls=-1)
    dataset = dataset.map(lambda x, y: augm_func(x, y, **preprocess_func_args), num_parallel_calls=-1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def build_tf_dataset(dataset_name, dataset_split, batch_size, num_of_classes, use_augmentation,
                     images_shuffle_size=None, shuffle_tfrecords=False, resize_shape=None):

    dataset, metadata = tfds.load(dataset_name, split=dataset_split, shuffle_files=shuffle_tfrecords,
                                  as_supervised=True, with_info=True)
    num_of_batches = (metadata.splits[dataset_split].num_examples + batch_size - 1) // batch_size

    dataset = dataset.shuffle(images_shuffle_size) if images_shuffle_size else dataset

    augmentation_args = {'resize_shape': resize_shape, 'num_of_classes': num_of_classes}

    # Choose between augmentation or simple scaling
    curr_func = augmentation_func if use_augmentation else preprocess_img_label

    dataset = dataset.map(lambda x, y: curr_func(x, y, **augmentation_args), num_parallel_calls=-1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    num_of_examples = metadata.splits[dataset_split].num_examples
    print(f"{dataset_name}-{dataset_split} set has {num_of_examples} examples")

    return dataset, metadata, num_of_batches


def build_custom_dataset(tfrecords_paths, percentage, set_name, use_augmentation, num_of_classes, batch_size,
                         images_per_tfrecord, images_shuffle_size=None, shuffle_tfrecords=True, return_metadata=False,
                         resize_shape=None):

    tfrecords_list, num_of_batches = choose_tfrecords(tfrecords_paths, percentage, set_name, batch_size,
                                                      images_per_tfrecord, shuffle_tfrecords)

    dataset = tf.data.TFRecordDataset(tfrecords_list, num_parallel_reads=-1)
    dataset = dataset.map(lambda x: _parse_image_function(x, return_metadata), num_parallel_calls=-1)
    dataset = dataset.shuffle(images_shuffle_size) if images_shuffle_size else dataset

    # Choose between augmentation or simple scaling
    curr_func = augmentation_func if use_augmentation else preprocess_img_label

    # Choose whether to return metadata or not
    args_with_md = {'return_metadata': return_metadata, 'resize_shape': resize_shape, 'num_of_classes': num_of_classes}
    simple_args = deepcopy(args_with_md)
    simple_args.pop('return_metadata')

    if return_metadata:
        dataset = dataset.map(lambda x, y, md: curr_func(x, y, md, **args_with_md), num_parallel_calls=-1)
    else:
        dataset = dataset.map(lambda x, y: curr_func(x, y, **simple_args), num_parallel_calls=-1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, num_of_batches

