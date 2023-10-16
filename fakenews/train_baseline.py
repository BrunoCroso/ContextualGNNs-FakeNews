#!/usr/bin/env python

import logging
import argparse
import jgrapht
import json
import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.transforms import NormalizeFeatures

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import tensorflow as tf


def tree_to_data(filename):
    logging.debug("Reading {}".format(filename))

    # read label
    with open(filename) as json_file:
        data = json.load(json_file)
        label = data["label"]
        # 0 = true, 1 = fake
        is_fake = label == "fake"

    vfeatures = []
    for node in data['nodes']:
        as_list = []
        for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count",
                    "favourites_count", "verified", ]:
            as_list.append(float(node[key]))
        as_list.extend(node["embedding"])
        vfeatures.append(as_list)

    #scaler = MinMaxScaler()
    #vfeatures = scaler.fit_transform(vfeatures)
    features = np.mean(vfeatures, axis=0)

    return label == "fake", features



from sklearn.metrics import confusion_matrix


#    pred = torch.cat(preds, dim=0).numpy()
#    test_f1 = f1_score(y, pred, average='binary')
#    test_precision = precision_score(y, pred)
#    test_recall = recall_score(y, pred)
#    test_accuracy = accuracy_score(y, pred)

#    print('y:##############')
#    print(y)
#    print('pred###############')
#    print(pred)

#    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
#    print('tn:%s fp:%s fn:%s tp:%s' % (tn, fp, fn, tp))
#    return test_f1, test_precision, test_recall, test_accuracy


from sklearn.utils import class_weight


def run(root_path):
    logging.info("Loading dags dataset")

    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'val')
    test_path = os.path.join(root_path, 'test')

    datasets = []
    labels = []
    for i, path in enumerate([train_path, val_path, test_path]):
        number_of_reals = 0
        dataset_fake = []
        dataset_real = []
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                label,  t = tree_to_data(fentry.path)
                number_of_features = t.shape[0]
                if label == 0:
                    dataset_real.append(t)
                else:
                    dataset_fake.append(t)

        number_of_samples = min(len(dataset_real), len(dataset_fake))
        number_of_samples = 5000000
        #number_of_real = len(dataset_real)
        #number_of_fake = len(dataset_fake)
        #if i == 0:
        #    multiply_by = number_of_real / number_of_fake
        #    print('multiply by')
        #    print(multiply_by)
        #    multiply_by = int(multiply_by)
        #    dataset_fake = dataset_fake * multiply_by
        dataset = dataset_real[:number_of_samples] + dataset_fake[:number_of_samples]
        labels_real = [0]*len(dataset_real[:number_of_samples])
        labels_fake = [1]*len(dataset_fake[:number_of_samples])
        labels.append(labels_real + labels_fake)
        print('number of samples')
        print(i, len(dataset))
        datasets.append(dataset)

    # 0 = true, 1 = fake
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                y=labels[0])

    train_labels = labels[0]
    val_labels = labels[1]
    test_labels = labels[2]

    logging.info('Train dataset size: %s ' % len(train_labels))
    logging.info('Validation dataset size: %s ' % len(val_labels))
    logging.info('Test dataset size: %s' % len(test_labels))

    print('Number of fake news in train set:%s Number of real news: %s' % (
    len([i for i in train_labels if i == 1]), len([i for i in train_labels if i == 0])))
    print('Number of fake news in val set:%s Number of real news: %s' % (
    len([i for i in val_labels if i == 1]), len([i for i in val_labels if i == 0])))
    print('Number of fake news in test set:%s Number of real news: %s' % (
    len([i for i in test_labels if i == 1]), len([i for i in test_labels if i == 0])))

    train_ds = tf.data.Dataset.from_tensor_slices((datasets[0], train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((datasets[1], val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((datasets[2], test_labels))
    train_ds.shuffle(buffer_size=len(datasets[0]))

    train_ds = train_ds.batch(32)

    val_ds = val_ds.batch(32)

    print("number of features", number_of_features)
    model = keras.models.Sequential()
    #model.add(keras.layers.Input( dtype='float32'))
    model.add(keras.layers.LayerNormalization(input_dim=number_of_features))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(keras.optimizers.SGD(learning_rate=0.001, momentum=0.5), 'binary_crossentropy', metrics=['accuracy', tf.metrics.TruePositives(), tf.metrics.FalsePositives(), tf.metrics.TrueNegatives(), tf.metrics.FalseNegatives()])
    model.fit(train_ds, epochs=20, validation_data=val_ds, shuffle=True)

    print()
    print()
    h = model.evaluate(test_ds.batch(32))
    print(h)
    _, accuracy, tp, fp, tn, fn = h
    try:
        precision = tp/(tp + fp)
    except ZeroDivisionError:
        precision = 'NaN'
    try:
        recall = tp/(tp + fn)
    except ZeroDivisionError:
        recall = 'NaN'
    try:
        f1 = 2*tp/(2*tp + fp + fn)
    except ZeroDivisionError:
        f1 = 'NaN'
    print(root_path)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} F1: {f1}')


from tensorflow.keras.layers.experimental.preprocessing import Normalization
def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    for path in [
        'produced_data/dataset0',
        'produced_data/dataset1',
        'produced_data/dataset2',
        'produced_data/dataset3',
        'produced_data/dataset4',
        'produced_data/dataset5',
        'produced_data/dataset6',
        'produced_data/dataset7',
        'produced_data/dataset8',
        'produced_data/dataset9',
    ]:
        run(path)
