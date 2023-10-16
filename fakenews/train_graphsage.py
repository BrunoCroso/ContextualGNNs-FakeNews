#!/usr/bin/env python

#
# See https://stellargraph.readthedocs.io/en/latest/demos/node-classification/graphsage-inductive-node-classification.html
#

# For the balance of the datasets check:
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
# https://stackoverflow.com/questions/44716150/how-can-i-assign-a-class-weight-in-keras-in-a-simple-way/44721883
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
# https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
# https://elitedatascience.com/imbalanced-classes

import argparse
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import layers, Model, optimizers
from sklearn import preprocessing, model_selection
import utils
import tensorflow
import os
import json
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def infer(embedding_model, vertices_df, edges_df, batch_size, num_samples):
    hold_out_graph = StellarGraph(vertices_df, edges_df, edge_type_default='follows', node_type_default='user')
    hold_out_gen = GraphSAGENodeGenerator(hold_out_graph, batch_size, num_samples)
    hold_out_gen = hold_out_gen.flow(vertices_df.index)
    if type(embedding_model) in [str]:
        embedding_model = tensorflow.keras.models.load_model(embedding_model)
    emb = embedding_model.predict(hold_out_gen)
    return emb

def run(args):
    user_labels_dir = 'produced_data/user_labels'

    train_edges_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "train_edges.pkl"))
    train_vertices_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "train_vertices.pkl"))
    train_vertices_df.drop(['id'], inplace=True, axis=1)
    #train_vertices_df[:] = 0.029722
    train_labels = utils.create_user_labels_df(list(train_vertices_df.index), user_labels_dir)['label']

    utils.write_object_to_pickle_file('train_labels.pkl', train_labels)

    print('###### Total train labels value counts')
    print(train_labels.value_counts())

    val_edges_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "val_edges.pkl"))
    val_vertices_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "val_vertices.pkl"))
    val_vertices_df.drop(['id'], inplace=True, axis=1)
    #val_vertices_df[:] = 0.029722
    val_labels = utils.create_user_labels_df(list(val_vertices_df.index), user_labels_dir)['label']

    utils.write_object_to_pickle_file('val_labels.pkl', val_labels)

    print('###### Total val labels value counts')
    print(val_labels.value_counts())

    test_edges_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "test_edges.pkl"))
    test_vertices_df = utils.read_pickle_from_file(os.path.join(args.dataset_root, "test_vertices.pkl"))
    test_vertices_df.drop(['id'], inplace=True, axis=1)
    #test_vertices_df[:] = 0.029722
    test_labels = utils.create_user_labels_df(list(test_vertices_df.index), user_labels_dir)['label']

    utils.write_object_to_pickle_file('test_labels.pkl', test_labels)

    print('###### Total test labels value counts')
    print(test_labels.value_counts())

    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_labels)
    # print('###### class weights ######')
    # weights_sklearn = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1], y=[i[0] for i in train_targets])
    # weights = {
    #     0: weights_sklearn[0],
    #     1: weights_sklearn[1]
    # }
    # print(weights)
    val_targets = target_encoding.transform(val_labels)
    test_targets = target_encoding.transform(test_labels)

    train_graph = StellarGraph(train_vertices_df, train_edges_df, edge_type_default='follows', node_type_default='user')
    print(train_graph.info())

    batch_size = 100
    num_samples = [15, 10]
    train_generator = GraphSAGENodeGenerator(train_graph, batch_size, num_samples)
    train_gen = train_generator.flow(list(train_labels.index), train_targets, shuffle=True)

    graphsage_model = GraphSAGE(
        layer_sizes=[32, 15], generator=train_generator, bias=True, dropout=0.5, activations=['relu', 'relu']
    )
    x_inp, x_out = graphsage_model.in_out_tensors()
    if train_targets.shape[1] == 1:
        print('###### binary classification problem')
        activation = "sigmoid"
        loss = "binary_crossentropy"
        metrics = ["accuracy", tensorflow.keras.metrics.Precision(name='precision'), tensorflow.keras.metrics.Recall(name='recall')]
    else:
        print('###### multi-classification problem')
        activation = "softmax"
        loss = "categorical_crossentropy"
        metrics = ["acc"]
    prediction = layers.Dense(units=train_targets.shape[1], activation=activation)(x_out)
    print('###### prediction shape')
    print(prediction.shape)
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        #optimizer=optimizers.Adam(lr=0.001),
        optimizer=optimizers.SGD(lr=0.001),
        loss=loss,
        metrics=metrics,
    )

    val_graph = StellarGraph(val_vertices_df, val_edges_df, edge_type_default='follows', node_type_default='user')
    val_generator = GraphSAGENodeGenerator(val_graph, batch_size, num_samples)
    val_gen = val_generator.flow(val_labels.index, val_targets)
    history = model.fit(
        #train_gen, epochs=args.epochs, validation_data=val_gen, verbose=1, shuffle=False, class_weight=weights
        train_gen, epochs=args.epochs, validation_data=val_gen, verbose=1, shuffle=False
    )

    test_graph = StellarGraph(test_vertices_df, test_edges_df, edge_type_default='follows', node_type_default='user')
    test_generator = GraphSAGENodeGenerator(test_graph, batch_size, num_samples)
    test_gen = test_generator.flow(test_labels.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print(test_labels.to_numpy())
    print(target_encoding.inverse_transform(model.predict(test_gen)))
    cm = confusion_matrix(test_labels.to_numpy(), target_encoding.inverse_transform(model.predict(test_gen)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non-astroturfer', 'astroturfer'])
    disp.plot()
    plt.show()

    embedding_model_filename = 'users_embedding_%s_%s_model' % (batch_size, '_'.join([str(i) for i in num_samples]))
    print('Saving embedding_model: %s' % (embedding_model_filename))
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    embedding_model.save(embedding_model_filename)
    
    print('Saving users embeddings lookup file')
    embeddings_lookup = {}
    for vertices, edges in [(train_vertices_df, train_edges_df), (val_vertices_df, val_edges_df), (test_vertices_df, test_edges_df)]:
        embeddings = infer(embedding_model, vertices, edges, batch_size, num_samples)
        for i, index in enumerate(list(vertices.index)):
            embeddings_lookup[index] = embeddings[i].tolist()
    with open(os.path.join(args.dataset_root, 'users_graphsage_embeddings_lookup.json'), 'w') as f:
        json.dump(embeddings_lookup, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Example: python train_graphsage.py"
    )
    parser.add_argument(
        "--dataset-root",
        help="Directory of the dataset",
        dest="dataset_root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        dest="epochs",
        type=int,
        default=5
    )

    args = parser.parse_args()
    run(args)
