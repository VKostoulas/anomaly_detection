import os
import numpy as np
import random
import glob
import math
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow as tf

# print(tfds.list_builders())


def convert_labels_to_zeros_and_ones(normal_class_label, labels):
    # change labels to 0 for normal and 1 for anomalies
    curr_labels = []
    for label in labels:
        if label == normal_class_label:
            curr_labels.append(0)
        else:
            curr_labels.append(1)
    labels = np.array(curr_labels)
    return labels


def extract_percentage_of_data(images, labels, class_percentage):

    class_labels = np.unique(labels)
    final_images, final_labels = [], []

    class_percentages = class_percentage.split(',') if ',' in class_percentage else [class_percentage]

    for class_percentage in class_percentages:
        curr_class_percentage = class_percentage.split(':')
        first_part_percentage = '0' if not curr_class_percentage[0] else curr_class_percentage[0]
        second_part_percentage = '100' if not curr_class_percentage[1] else curr_class_percentage[1]
        first_part_percentage = float(first_part_percentage) / 100
        second_part_percentage = float(second_part_percentage) / 100

        for class_label in class_labels:
            curr_class_mask = np.where(labels == class_label, True, False).tolist()
            curr_class_mask = sum(curr_class_mask, []) if isinstance(curr_class_mask[0], list) else curr_class_mask
            curr_class_labels = labels[curr_class_mask]
            curr_class_images = images[curr_class_mask]
            num_of_curr_class_examples = curr_class_images.shape[0]

            start_index = int(np.floor(first_part_percentage * num_of_curr_class_examples))
            end_index = int(np.floor(second_part_percentage * num_of_curr_class_examples))

            final_images.append(curr_class_images[start_index: end_index])
            final_labels.append(curr_class_labels[start_index: end_index])

    final_images = final_images[0] if len(final_images) == 1 else np.concatenate(final_images, axis=0)
    final_labels = final_labels[0] if len(final_labels) == 1 else np.concatenate(final_labels, axis=0)

    return final_images, final_labels


def select_examples_from_label(normal_class_label, anomaly_class_labels, images, labels):

    selected_indexes = []
    for i, label in enumerate(labels):
        if label == normal_class_label or label in anomaly_class_labels:
            selected_indexes.append(i)

    final_images = images[selected_indexes]
    final_labels = labels[selected_indexes]

    return final_images, final_labels


def choose_tfrecords(tfrecords_paths, percentage, set_name, batch_size, images_per_tfrecord, shuffle_tfrecords):

    files = sorted(list(glob.glob(tfrecords_paths)))
    files = [path.replace('\\', '/') for path in files if '\\' in path]
    min_files = 1
    if len(files) == 0:
        raise ValueError('There are no tfrecords in %s' % tfrecords_paths)
    num_files = max(int(len(files) * percentage), min_files)

    set_size = num_files * images_per_tfrecord
    set_batch_num = (set_size + batch_size - 1) // batch_size

    if shuffle_tfrecords:
        np.random.shuffle(files)

    print('%-5s set has approximately %d examples' % (set_name, set_size))
    return files[:num_files], set_batch_num


def preprocess_img_label(image, label, metadata=None, return_metadata=False, num_of_classes=2,
                         resize_shape=None, preprocess_mode='-1,1'):
    image = tf.cast(image, tf.float32)

    if preprocess_mode == '0,1':
        image /= 255.
        label = tf.one_hot(label, num_of_classes)
    elif preprocess_mode == 'convert':
        image *= 2
        image -= 1.
    elif preprocess_mode == '-1,1':
        image /= 127.5
        image -= 1.
        label = tf.one_hot(label, num_of_classes)
    else:
        raise ValueError('preprocess_mode does not exist')

    if len(image.shape) == 2:
        image = tf.concat(
            [tf.expand_dims(image, axis=-1), tf.expand_dims(image, axis=-1), tf.expand_dims(image, axis=-1)],
            axis=-1)

    if resize_shape:
        image = tf.image.resize(image, resize_shape)

    if return_metadata:
        output = image, label, metadata
    else:
        output = image, label

    return output


def augmentation_func(image, label, metadata=None, return_metadata=False, resize_shape=None, num_of_classes=2, prob=0.5):

    # Images must be in 0,1 range before augmentation
    scale_args_with_md = [image, label, metadata, return_metadata, num_of_classes, resize_shape, '0,1']
    scale_args_without_md = [image, label, None, False, num_of_classes, resize_shape, '0,1']
    curr_scale_args = scale_args_with_md if return_metadata else scale_args_without_md

    output = preprocess_img_label(*curr_scale_args)

    output = augment_image(output, prob)

    # Return back images to -1,1 range  after augmentation
    convert_args_with_md = [*output, return_metadata, num_of_classes, None,  'convert']
    convert_args_without_md = [*output, None, False, num_of_classes, None, 'convert']
    curr_convert_args = convert_args_with_md if return_metadata else convert_args_without_md

    # Output has length 2 or 3 (without or with metadata respectively)
    output = preprocess_img_label(*curr_convert_args)
    return output


def augment_image(output, prob):
    # output is image-label or image-label-metadata
    image, *_ = output

    if tf.random.uniform(shape=[1, 1])[0][0] <= prob:
        image = tf.image.random_flip_left_right(image)
    # if tf.random.uniform(shape=[1, 1])[0][0] <= prob:
    #     image = tf.image.random_flip_up_down(image)
    # if tf.random.uniform(shape=[1, 1])[0][0] <= prob:
    #     # angle = tf.random.uniform(shape=[1, 1])[0][0] * 360
    #     # image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    #     random_rotation_index = tf.random.uniform(shape=[1], minval=1, maxval=4, dtype=tf.int32)[0]
    #     image = tf.image.rot90(image, k=random_rotation_index)
    if tf.random.uniform(shape=[1, 1])[0][0] <= prob:
        image = tf.image.adjust_saturation(image, 2)
    if tf.random.uniform(shape=[1, 1])[0][0] <= prob:
        image = tf.image.adjust_brightness(image, 0.3)
    # image = tf.image.rgb_to_grayscale(image)
    # image = tf.concat([image, image, image], axis=-1)

    return (image,) + output[1:]


def show_an_image(dataset, index=-1):
    images, labels = next(iter(dataset))
    if index == -1:
        random_index = random.randrange(images.shape[0])
        image = images[random_index]
        label = labels[random_index]
    else:
        image = images[index]
        label = labels[index]
    label = [index for index in range(len(label)) if label[index] == 1.0][0]
    # Image is in -1,1 range so convert it to 0,255 integers
    image += 1.0
    image *= 127.5
    image = image.numpy().astype(int)
    plt.imshow(image)
    plt.title(f"Image Shape: {image.shape}, Label: {label}")
    plt.show()
    plt.close()


def _parse_image_function(example_proto, return_metadata):
    # Create a dictionary describing the features.
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_jpeg(features['image'], channels=3)
    label = features['label']
    if return_metadata:
        metadata = features['id']
        output = image, label, metadata
    else:
        output = image, label

    return output


def create_tfrecords_from_tfdatasets(dataset_name, dataset_split, classes_dict, save_path):

    # dataset now is a list (maybe cause dataset split argument is a list)
    dataset, metadata = tfds.load(dataset_name, split=dataset_split, shuffle_files=True,
                                  as_supervised=True, with_info=True)

    # num_examples_in_datasets = [metadata._splits[name].num_examples for name in dataset_split]

    # Initialize images and tfrecords for each class
    class_images = {}
    class_num_of_tfrecords_created = {}
    class_curr_tfrecord_name = {}
    class_save_path = {}
    class_curr_save_tfrecord_path = {}
    class_curr_tfrecord_writer = {}

    for class_name in classes_dict:
        class_images[class_name] = 0
        class_num_of_tfrecords_created[class_name] = 0
        class_curr_tfrecord_name[class_name] = f"{dataset_split[0]}_" \
                                               f"{class_num_of_tfrecords_created[class_name]:04d}.tfrecord"
        class_save_path[class_name] = save_path + class_name + '/'
        if not os.path.exists(class_save_path[class_name]):
            os.makedirs(class_save_path[class_name])
        class_curr_save_tfrecord_path[class_name] = class_save_path[class_name] + class_curr_tfrecord_name[class_name]
        class_curr_tfrecord_writer[class_name] = tf.io.TFRecordWriter(class_curr_save_tfrecord_path[class_name])

    # Iterate over the tfdataset
    for i, (image, label) in enumerate(dataset):
        img_name = 'image-' + str(i) + '_label-' + str(label.numpy())
        img_name = img_name.encode()
        image_encoded = tf.image.encode_jpeg(image).numpy()
        label = label.numpy()

        # Separate and save images per class
        for class_name in classes_dict:
            if label == classes_dict[class_name]:

                class_images[class_name] += 1

                if class_images[class_name] % 5001 == 0:
                    class_num_of_tfrecords_created[class_name] += 1
                    class_curr_tfrecord_name[class_name] = f"{dataset_split[0]}_" \
                                                           f"{class_num_of_tfrecords_created[class_name]:04d}.tfrecord"
                    class_curr_save_tfrecord_path[class_name] = class_save_path[class_name] + class_curr_tfrecord_name[
                        class_name]

                    class_curr_tfrecord_writer[class_name].close()
                    class_curr_tfrecord_writer[class_name] = tf.io.TFRecordWriter(
                        class_curr_save_tfrecord_path[class_name])

                # Feature contains a map of string to feature proto objects
                features_dict = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),
                                 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                                 'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name]))
                                 }
                # Construct the Example proto object
                example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                # Serialize the example to a string
                serialized = example.SerializeToString()

                class_curr_tfrecord_writer[class_name].write(serialized)

    for class_name in classes_dict:
        class_curr_tfrecord_writer[class_name].close()

