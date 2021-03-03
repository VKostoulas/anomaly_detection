"""
Implements train loop.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def model_infer_loop(model, data, loss_func, title_name, num_of_samples, show_hist):

    anomaly_scores = []
    all_labels = []
    data_gen, num_of_batches = extract_data(data)
    inf_tqdm = tqdm(data_gen, total=num_of_batches, leave=False, desc='Inference', ascii=True, file=sys.stdout)

    for (images, labels) in inf_tqdm:

        output_feature_vectors = model(images, training=False)
        image_anomaly_scores = loss_func(y_true=None, y_pred=output_feature_vectors)
        image_anomaly_scores, labels = post_process_data(image_anomaly_scores, labels)

        anomaly_scores.extend(image_anomaly_scores)
        all_labels.extend(labels)

        if num_of_samples and len(anomaly_scores) >= num_of_samples:
            anomaly_scores = anomaly_scores[:num_of_samples]
            all_labels = all_labels[:num_of_samples]
            break

    _ = inference_results(anomaly_scores, all_labels, title_name, show_hist=show_hist)


def extract_data(data):
    """
    Extract data generator and number of batches according to he type of data object.

    :param data:
    :return:
    """
    if isinstance(data, dict):
        if len(list(data.keys())) == 3:
            data_gen, num_of_batches, _ = data.values()
        else:
            data_gen, num_of_batches = data.values()
    elif isinstance(data, tuple):
        if len(data) == 3:
            data_gen, num_of_batches, _ = data
        else:
            data_gen, num_of_batches = data
    else:
        raise ValueError(f"data type is not valid ({type(data)})")
    return data_gen, num_of_batches


def inference_results(anomaly_scores, labels, title_name, show_hist, verbose=1):

    # separate tumor and normal scores for validation set to create 2 colored histograms
    class_0_scores = [score for score, label in zip(anomaly_scores, labels) if label == 0.0]
    class_1_scores = [score for score, label in zip(anomaly_scores, labels) if label == 1.0]
    curr_anomaly_scores = [class_0_scores, class_1_scores]

    create_histogram(curr_anomaly_scores, density=False, facecolor='g', alpha=0.75, title=title_name,
                     x_label='Anomaly Scores ({} samples)'.format(len(anomaly_scores)), y_label='Frequency',
                     grid=False, show_histograms_plots=show_hist)

    mean_value, std_value = np.mean(anomaly_scores), np.std(anomaly_scores)
    roc_auc = roc_auc_score(labels, anomaly_scores)
    if verbose > 0:
        print(f'\nROC-AUC Score: {roc_auc:.4f}, Mean Anomaly Score: {mean_value:.4f}, Std: {std_value:.4f}')
    return roc_auc


def post_process_data(image_anomaly_scores, labels):

    image_anomaly_scores = image_anomaly_scores.numpy().tolist()
    labels = labels.numpy().tolist()

    for i, one_hot_label in enumerate(labels):
        if not isinstance(one_hot_label, list):
            raise ValueError(f'Labels should contain one hot encoded labels in lists, but {one_hot_label} was given')
        else:
            # Convert back to scalar from one hot encoding
            labels[i] = [index for index, label in enumerate(one_hot_label) if label == 1.0][0]

    return image_anomaly_scores, labels


def create_histogram(values, density=None, facecolor=None, alpha=None, title=None,
                     x_label=None, y_label=None, grid=None, show_histograms_plots=False):
    if show_histograms_plots:
        # show matplotlib histogram
        if isinstance(values[0], list):
            # values contain two lists, anomaly scores for normal and tumor patches respectively
            plt.hist(values[0], bins=50, label='normal', density=density, alpha=alpha)
            plt.hist(values[1], bins=50, label='tumor', density=density, alpha=alpha)
            plt.legend(loc='upper right')
        else:
            plt.hist(values, bins='auto', density=density, facecolor=facecolor, alpha=alpha)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(grid)

        plt.show()
        plt.close()
