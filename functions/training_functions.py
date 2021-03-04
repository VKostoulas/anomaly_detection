import os
import numpy as np
from datetime import datetime

from functions.anomaly_models_functions import build_anomaly_detection_model, build_autoencoder, build_classifier, \
    PerceptualAutoEncoder, define_image_size, PerceptualEnsemble, PerceptualMultiClassifier, inference_on_test_set
from functions.common_functions import define_mixed_precision_policy, define_callbacks, define_optimizer, \
    define_loss_function, DisplayCallback, CustomInference2, extract_best_weights, create_model_path, \
    calculate_class_percentage
from functions.gans_functions import GAN, CustomLRScheduler
from functions.build_datasets import choose_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.callbacks import ReduceLROnPlateau


def train_simple_classifier(s, model, train_data, val_data):

    define_mixed_precision_policy(s)

    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=define_optimizer(s.OPTIMIZER, s.OPTIMIZER_PARAMS),
        metrics=['categorical_accuracy', 'AUC'],
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-7)

    model.fit(x=train_data, epochs=s.EPOCHS, validation_data=val_data, callbacks=[reduce_lr])


def train_anomaly_detection_model(s):
    define_mixed_precision_policy(s.USE_MIXED_PRECISION)
    train_dataset, val_dataset, test_dataset = choose_dataset(s)
    image_size = define_image_size(s)
    auto_encoder_model = build_autoencoder(s.AUTOENCODER_ARCHITECTURE, image_size, s.AUTOENCODER_PARAMS)
    classifier_model = build_classifier(architecture=s.CLASSIFIER_ARCHITECTURE, loss_layers=s.PERCEPTUAL_LOSS_LAYERS,
                                        image_size=image_size, clf_kwargs=s.CLASSIFIER_PARAMS)
    model = build_anomaly_detection_model(s=s, auto_encoder=auto_encoder_model, classifier=classifier_model, show_graph=True)
    model_path = create_model_path(s)
    callbacks = define_callbacks(s=s, val_dataset=val_dataset, auto_encoder=auto_encoder_model, model_path=model_path)
    model.fit(x=train_dataset, validation_data=val_dataset, epochs=s.EPOCHS, callbacks=callbacks)
    _ = inference_on_test_set(s, model, test_dataset, model_path)


def k_fold_training(s):

    define_mixed_precision_policy(s.USE_MIXED_PRECISION)
    k_folds = s.K_FOLD_NUM
    class_print_strings = []
    average_score = []

    for curr_class_label in range(s.CLASSES):

        normal_class = curr_class_label
        anomaly_classes = [np.random.randint(0, s.CLASSES)]
        while anomaly_classes[0] == normal_class:
            anomaly_classes = [np.random.randint(0, s.CLASSES)]

        curr_class_test_auc = []

        for k_step in np.arange(0, 100, 100/k_folds):

            train_class_percentage, val_class_percentage = calculate_class_percentage(k_folds, k_step)
            train_dataset, val_dataset, test_dataset = choose_dataset(s=s,
                                                                      train_class_percentage=train_class_percentage,
                                                                      val_class_percentage=val_class_percentage,
                                                                      normal_class=normal_class,
                                                                      anomaly_classes=anomaly_classes)
            image_size = define_image_size(s)
            auto_encoder_model = build_autoencoder(s.AUTOENCODER_ARCHITECTURE, image_size, s.AUTOENCODER_PARAMS)
            classifier_model = build_classifier(architecture=s.CLASSIFIER_ARCHITECTURE,
                                                loss_layers=s.PERCEPTUAL_LOSS_LAYERS,
                                                image_size=image_size, clf_kwargs=s.CLASSIFIER_PARAMS)
            anomaly_detector = build_anomaly_detection_model(s, auto_encoder_model, classifier_model, show_graph=True)
            model_path = create_model_path(s)
            callbacks = define_callbacks(s, val_dataset, auto_encoder_model, model_path)
            anomaly_detector.fit(train_dataset, validation_data=val_dataset, epochs=s.EPOCHS, callbacks=callbacks)
            model_path = inference_on_test_set(s, anomaly_detector, test_dataset, model_path)
            *_, auc_on_test_set = extract_best_weights(model_path)
            curr_class_test_auc.append(auc_on_test_set * 100)

        print_string = f'AUC for class {curr_class_label} on test set: ' \
                       f'{np.mean(curr_class_test_auc):.4f} +/- ' \
                       f'{np.std(curr_class_test_auc):.4f}'
        average_score.append(np.mean(curr_class_test_auc))
        class_print_strings.append(print_string)

    print('\n')
    for print_string in class_print_strings:
        print(print_string)
    print(f'\nAverage Score: {np.mean(average_score)}')


def gans_training(s):
    train_dataset, val_dataset, test_dataset = choose_dataset(s=s,
                                                              train_class_percentage=':90',
                                                              val_class_percentage='90:',
                                                              normal_class=0,
                                                              anomaly_classes=[5])

    image_size = define_image_size(s)
    auto_encoder = build_autoencoder(s.AUTOENCODER_ARCHITECTURE, image_size, s.AUTOENCODER_PARAMS)
    classifier = build_classifier(architecture=s.CLASSIFIER_ARCHITECTURE, loss_layers=s.PERCEPTUAL_LOSS_LAYERS,
                                  image_size=image_size, clf_kwargs=s.CLASSIFIER_PARAMS)

    # add dense layers to classifier
    gap_layer = tf.keras.layers.GlobalAvgPool2D()(classifier.output)
    # drop_out_layer = tf.keras.layers.Dropout(0.4)(gap_layer)
    final_dense_layer = tf.keras.layers.Dense(1)(gap_layer)
    classifier_outputs = [classifier.output, final_dense_layer]
    classifier = tf.keras.Model(inputs=classifier.input, outputs=classifier_outputs)

    auto_encoder.summary()
    classifier.summary()

    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    clf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    gan = GAN(s, auto_encoder, classifier)
    gan.compile(
        ae_optimizer=ae_optimizer,
        clf_optimizer=clf_optimizer,
        clf_loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
        ae_loss_fn=define_loss_function(s.LOSS_FUNCTION),
    )

    callbacks = []
    if s.SHOW_RECONSTRUCTIONS:
        display_cb = DisplayCallback(data=val_dataset, infer_model=auto_encoder, num_of_samples=2, sample_indexes=None)
        callbacks.append(display_cb)

    model_save_time = datetime.today().strftime('%Y-%m-%d') + '_' + datetime.now().time().strftime('%H-%M-%S')
    model_path = s.PROJECT_MODEL_PATH + model_save_time
    inference_cb = CustomInference2(val_data=val_dataset, save_model_path=model_path)
    callbacks.append(inference_cb)

    def lr_time_based_decay(epoch, learning_rate, decay=0.01):
        return learning_rate * (1 / (1 + decay * epoch))

    ae_lr_cb = CustomLRScheduler(schedule=lr_time_based_decay, optimizer_mode='ae_optimizer', verbose=1)
    clf_lr_cb = CustomLRScheduler(schedule=lr_time_based_decay, optimizer_mode='clf_optimizer', verbose=1)
    callbacks.extend([ae_lr_cb, clf_lr_cb])

    gan.fit(train_dataset, validation_data=val_dataset, epochs=s.EPOCHS, callbacks=callbacks)

    # restore weights
    best_ae_weights, best_clf_weights, *_ = extract_best_weights(model_path)
    gan.autoencoder.load_weights(best_ae_weights)
    gan.classifier.load_weights(best_clf_weights)

    cb = CustomInference2(val_data=test_dataset, save_model_path=None)
    gan.evaluate(test_dataset, callbacks=[cb])


def experimental_training(s, mode, data_params, ae_params=None, clf_params=None, global_params=None):

    for attribute in global_params:
        setattr(s, attribute, global_params[attribute])

    define_mixed_precision_policy(s.USE_MIXED_PRECISION)
    train_dataset, val_dataset, test_dataset = choose_dataset(**data_params)

    if mode == 'different_losses':
        auto_encoder_model = build_autoencoder(s.AUTOENCODER_ARCHITECTURE, define_image_size(s), s.AUTOENCODER_PARAMS)
        classifier_model = build_classifier(architecture=s.CLASSIFIER_ARCHITECTURE,
                                            loss_layers=s.PERCEPTUAL_LOSS_LAYERS,
                                            image_size=define_image_size(s), clf_kwargs=s.CLASSIFIER_PARAMS)
        model = build_anomaly_detection_model(s=s, auto_encoder=auto_encoder_model, classifier=classifier_model)
        model_path = create_model_path(s)
        callbacks = define_callbacks(s, val_dataset, auto_encoder_model, model_path, verbose=0)
        model.fit(x=train_dataset, validation_data=val_dataset, verbose=0, epochs=s.EPOCHS, callbacks=callbacks)
        _ = inference_on_test_set(s, model, test_dataset, model_path, verbose=0)

    if mode == 'multi_classifier':

        image_size = define_image_size(s)
        autoencoder = build_autoencoder(s.AUTOENCODER_ARCHITECTURE, image_size, s.s.AUTOENCODER_PARAMS)
        classifiers = []
        for params in clf_params:
            classifier = build_classifier(**params)
            classifiers.append(classifier)

        multi_classifier_training(s, autoencoder, classifiers, train_dataset, val_dataset, test_dataset)

    elif mode == 'perceptual_ensemble':

        anomaly_models = []
        model_paths = []

        for params1, params2 in zip(ae_params, clf_params):
            autoencoder = build_autoencoder(**params1)
            classifier = build_classifier(**params2)
            anomaly_detector = build_anomaly_detection_model(s, autoencoder, classifier)
            anomaly_models.append(anomaly_detector)
            model_path = create_model_path(s)
            model_paths.append(model_path)
            callbacks = define_callbacks(s, val_dataset, autoencoder, model_path)
            anomaly_detector.fit(train_dataset, validation_data=val_dataset, epochs=s.EPOCHS, callbacks=callbacks)
            # restore weights
            best_ae_weights, best_clf_weights, _ = extract_best_weights(model_path)
            anomaly_detector.autoencoder.load_weights(best_ae_weights)
            anomaly_detector.classifier.load_weights(best_clf_weights)
            cb = CustomInference2(val_data=test_dataset, save_model_path=None, show_hist=s.SHOW_HISTOGRAMS_PLOTS)
            anomaly_detector.evaluate(test_dataset, callbacks=[cb])
        # inference on test set with ensemble of best models
        anomaly_detector = PerceptualEnsemble(*anomaly_models)
        anomaly_detector.compile()
        cb = CustomInference2(val_data=test_dataset, save_model_path=None, show_hist=s.SHOW_HISTOGRAMS_PLOTS)
        anomaly_detector.evaluate(test_dataset, callbacks=[cb])


def multi_classifier_training(s, autoencoder, classifiers, train_dataset, val_dataset, test_dataset):

    anomaly_detector = PerceptualMultiClassifier(autoencoder, *classifiers)
    loss_function = define_loss_function(s.LOSS_FUNCTION)
    optimizer = define_optimizer(s.OPTIMIZER, s.OPTIMIZER_PARAMS)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    anomaly_detector.compile(optimizer=optimizer, loss=loss_function)
    model_save_time = datetime.today().strftime('%Y-%m-%d') + '_' + datetime.now().time().strftime('%H-%M-%S')
    model_path = s.PROJECT_MODEL_PATH + model_save_time
    inference_cb = CustomInference2(val_data=val_dataset, save_model_path=model_path, show_hist=s.SHOW_HISTOGRAMS_PLOTS)
    anomaly_detector.fit(train_dataset, validation_data=val_dataset, epochs=s.EPOCHS, callbacks=[inference_cb])

    # restore weights
    best_ae_weights, best_clf1_weights, best_clf2_weights, _ = extract_best_weights(model_path)
    anomaly_detector.autoencoder.load_weights(best_ae_weights)
    anomaly_detector.classifier1.load_weights(best_clf1_weights)
    anomaly_detector.classifier2.load_weights(best_clf2_weights)

    cb = CustomInference2(val_data=test_dataset, save_model_path=None, show_hist=s.SHOW_HISTOGRAMS_PLOTS)
    anomaly_detector.evaluate(test_dataset, callbacks=[cb])
