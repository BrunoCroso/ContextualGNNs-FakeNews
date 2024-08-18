#!/usr/bin/env python

'''
Trains and evaluates a GAT (Graph Attention Network) model on the given dataset.
'''

import logging
import argparse
import jgrapht
import json
import os
import random
import pandas as pd
import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import brier_score_loss



from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm


def tree_to_data(filename):
    '''
    This function is responsible for processing and converting a JSON file representing news data propagation (in tree format)
    into a format suitable for  training a Graph Attention Network (GAT) model. The function extracts relevant features, label
    information, and constructs  a PyTorch Geometric Data object for graph-based machine learning.
    This function considers the specifications defined in options.json.
    '''

    logging.debug("Reading {}".format(filename))

    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    # read label
    with open(filename) as json_file:
        try:
            data = json.load(json_file)
        except:
            return None, None, None
        label = data["label"]
        # 0 = true, 1 = fake
        is_fake = label == "fake"

    vfeatures = []

    if options["user_embeddings"] == True and options["retweet_embeddings"] == True:
        for node in data['nodes']:
            as_list = []
            for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified",]:
                as_list.append(float(node[key]))
            as_list.extend(node["user_embedding"])
            as_list.extend(node["retweet_embedding"])
            vfeatures.append(as_list)

    if options["user_embeddings"] == True and options["retweet_embeddings"] == False:
        for node in data['nodes']:
            as_list = []
            for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified",]:
                as_list.append(float(node[key]))
            as_list.extend(node["user_embedding"])
            vfeatures.append(as_list)

    if options["user_embeddings"] == False and options["retweet_embeddings"] == True:
        for node in data['nodes']:
            as_list = []
            for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified",]:
                as_list.append(float(node[key]))
            as_list.extend(node["retweet_embedding"])
            vfeatures.append(as_list)

    if options["user_embeddings"] == False and options["retweet_embeddings"] == False:
        for node in data['nodes']:
            as_list = []
            for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified",]:
                as_list.append(float(node[key]))
            vfeatures.append(as_list)

    vlabels = []
    vlabels.append(is_fake)

    edge_sources = []
    edge_targets = []
    for e in data['edges']:
        edge_sources.append(e['source'])
        edge_targets.append(e['target'])

    x = torch.tensor(vfeatures, dtype=torch.float)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    y = torch.tensor(vlabels, dtype=torch.long)
    result = Data(x=x, edge_index=edge_index, y=y)
    #result = NormalizeFeatures()(result)

    number_of_features = len(vfeatures[0])
    #print(number_of_features)
    return label, number_of_features, result


from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.norm import GraphSizeNorm

class Net(torch.nn.Module):
    '''
    This class defines a Graph Attention Network (GAT) model
    '''

    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.batch_norm = BatchNorm(num_features)
        self.conv1 = GATConv(num_features, 32, heads=4, dropout=0.5)
        self.conv2 = GATConv(32 * 4, num_classes, heads=1, concat=False, dropout=0.1)


    def forward(self, data):
        '''
        Defines the forward pass of the GAT model.
        '''

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.batch_norm(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for torch.nn.Embedding
    layers. This method is slightly adapted from the original source code that can be found here:
    https://github.com/neelsjain/NEFTune Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```
    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha` to
            the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    
    if module.training:
        try:
            sparse_features = output[:, :8]
            dense_features = output[:, 8:]

            if dense_features.size(1) > 0:
                dims = torch.tensor(dense_features.size(1))
                mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
                dense_features = dense_features + torch.zeros_like(dense_features).uniform_(-mag_norm, mag_norm)
                output = torch.cat((sparse_features, dense_features), dim=1)

            else:
                return output
        except:
            print('ERROR')
            print(output[0].shape)
    
    return output

def train(model, loader, device, optimizer, loss_op):
    '''
    Performs a training loop on the GAT (Graph Attention Network) model on a graph dataset.
    '''

    model.train()

    total_loss = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        y_true = data.y
        #import pdb; pdb.set_trace()
        loss = loss_op(y_pred, y_true)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


    print('loader.dataset')
    print(f'Dataset size: {len(loader.dataset)}')
    return total_loss / len(loader.dataset)

from sklearn.metrics import confusion_matrix

@torch.no_grad()
def test(model, loader, device, loss_op):
    '''
    Evaluates the performance of the GAT (Graph Attention Network) model on the dataset using the test data loader.
    '''

    model.eval()

    total_loss = 0
    ys, preds = [], []

    for data in loader:
        data = data.to(device)
        y_true = data.y
        y_pred = model(data)
        loss = loss_op(y_pred, y_true)
        total_loss += loss.item() * data.num_graphs

        ys.append(data.y)
        out = model(data.to(device))
        preds.append(torch.argmax(out, dim=1).cpu())

    y = torch.cat(ys, dim=0).numpy()
    pred = torch.cat(preds, dim=0).numpy()

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
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

    
    
    accuracy = accuracy_score(y, pred)

    final_loss = total_loss / len(loader.dataset)

    #print('y:##############')
    #print(y)
    #print('pred###############')
    #print(pred)

    roc_auc = roc_auc_score(y, pred)

    precision_vector, recall_vector, thresholds_vector = precision_recall_curve(y, pred)

    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall_vector, precision_vector)

    geom_mean_score = geometric_mean_score(y, pred)

    brier_score = brier_score_loss(y, pred)

    f1_macro = f1_score(y, pred, average='macro')
    f1_micro = f1_score(y, pred, average='micro')
    f1_weighted = f1_score(y, pred, average='weighted')
    f1_binary = f1_score(y, pred, average='binary')

    #print('tn:%s fp:%s fn:%s tp:%s' % (tn, fp, fn, tp))
    return f1, precision, recall, accuracy, final_loss, roc_auc, auc_precision_recall, geom_mean_score, brier_score, f1_macro, f1_micro, f1_weighted, f1_binary

from sklearn.utils import class_weight

def run(root_path, neftune_noise_alpha):
    '''
    Trains and evaluates a GAT (Graph Attention Network) model on the given dataset.
    '''

    logging.info("Loading dags dataset")

    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'val')
    test_path = os.path.join(root_path, 'test')

    datasets = [] # It will contain 3 sublists: train, val and test

    for i, path in enumerate([train_path, val_path, test_path]):
        dataset_fake = []
        dataset_real = []
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                label, number_of_features, t = tree_to_data(fentry.path)
                if label == "real":
                    dataset_real.append(t)
                elif label == "fake":
                    dataset_fake.append(t)



        #This part was used to balance the train dataset (upsampling)
        if path == train_path:

            # Upsampling
            
            number_of_real = len(dataset_real)
            number_of_fake = len(dataset_fake)

            multiply_by = int(number_of_real / number_of_fake)
            dataset_fake = dataset_fake * multiply_by

            number_of_samples = min(len(dataset_real), len(dataset_fake)) 
            dataset = dataset_real[:number_of_samples] + dataset_fake[:number_of_samples] 

            # Unbalanced
            '''
            dataset = dataset_real + dataset_fake'''

            # Downsampling
            '''
            number_of_real = len(dataset_real)
            number_of_fake = len(dataset_fake)

            number_of_samples_downsampling = min(number_of_real, number_of_fake)

            random.shuffle(dataset_real)
            random.shuffle(dataset_fake)

            dataset = dataset_real[:number_of_samples_downsampling] + dataset_fake[:number_of_samples_downsampling]'''

        else:
            dataset = dataset_real + dataset_fake

        datasets.append(dataset)

    train_labels = [i.y.item() for i in datasets[0]]

    # Convertendo os rótulos para uma representação sequencial começando em 0
    train_labels_seq = [int(label) - min(train_labels) for label in train_labels]

    # Ordenando os rótulos sequenciais
    train_labels_seq.sort()

    weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                        y=train_labels_seq)
    
    val_labels = [i.y.item() for i in datasets[1]]
    test_labels = [i.y.item() for i in datasets[2]]

    logging.info('Train dataset size: %s ' % len(train_labels))
    logging.info('Validation dataset size: %s ' % len(val_labels))
    logging.info('Test dataset size: %s' % len(test_labels))

    print('Number of fake news in train set:%s Number of real news: %s' % (len([i for i in train_labels if i == 1]), len([i for i in train_labels if i == 0])))
    print('Number of fake news in val set:%s Number of real news: %s' % (len([i for i in val_labels if i == 1]), len([i for i in val_labels if i == 0])))
    print('Number of fake news in test set:%s Number of real news: %s' % (len([i for i in test_labels if i == 1]), len([i for i in test_labels if i == 0])))

    train_loader = DataLoader(datasets[0], batch_size=32, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=4, shuffle=True)
    test_loader = DataLoader(datasets[2], batch_size=4, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=number_of_features, num_classes=2).to(device)

    # Neftune parameters
    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    model.batch_norm.neftune_noise_alpha = neftune_noise_alpha
    model.batch_norm.register_forward_hook(neftune_post_forward_hook)

    # Teste
    #weights = [0.5, 4.5]
    #print('weights!!')

    
    #print('Class weights:', weights)
    #class_weights = torch.FloatTensor(weights).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    #loss_op = torch.nn.NLLLoss(weight=class_weights)
    

    # Descomentar esse trecho se precisar
    loss_op = torch.nn.NLLLoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    # Creating variables to save the best model
    best_model = None
    best_loss = None
    best_PR_AUC = None
    best_brier_score = None
    best_epoch = 1

    # Start training
    for epoch in range(1, 61):
        logging.info("Starting epoch {}".format(epoch))
        loss = train(model, train_loader, device, optimizer, loss_op)
        print(f'Loss: {loss}')

        # Checking the metrics obtained in the training set
        train_f1, train_precision, train_recall, train_accuracy, train_loss, train_roc_auc, train_auc_precision_recall, train_geometric_mean_score, train_brier_score_loss, train_f1_macro, train_f1_micro, train_f1_weighted, train_f1_binary = test(model, train_loader, device, loss_op)
        print(f'Metrics in training dataset:')
        print(f'Accuracy: {round(train_accuracy, 4)}, \tPrecision: {round(train_precision, 4)}, \tRecall: {round(train_recall, 4)}, \tF1: {round(train_f1, 4)}, \t\tROC AUC: {round(train_roc_auc, 4)}, \tPrecision-Recall AUC: {round(train_auc_precision_recall, 4)}')
        print(f'G-Means: {round(train_geometric_mean_score, 4)}, \tBrier Score: {round(train_brier_score_loss, 4)}, \tf1_macro: {round(train_f1_macro, 4)}, \tf1_micro: {round(train_f1_micro, 4)}, \tf1_weighted: {round(train_f1_weighted, 4)}, \tf1_binary: {round(train_f1_binary, 4)}')

        # Testing the model in the validation set
        val_f1, val_precision, val_recall, val_accuracy, val_loss, val_roc_auc, val_auc_precision_recall, val_geometric_mean_score, val_brier_score_loss, val_f1_macro, val_f1_micro, val_f1_weighted, val_f1_binary = test(model, val_loader, device, loss_op)
        print()
        print(f'Metrics in validation dataset:')
        print(f'Accuracy: {round(val_accuracy, 4)}, \tPrecision: {round(val_precision, 4)}, \tRecall: {round(val_recall, 4)}, \tF1: {round(val_f1, 4)}, \t\tROC AUC: {round(val_roc_auc, 4)}, \tPrecision-Recall AUC: {round(val_auc_precision_recall, 4)}')
        print(f'G-Means: {round(val_geometric_mean_score, 4)}, \tBrier Score: {round(val_brier_score_loss, 4)}, \tf1_macro: {round(val_f1_macro, 4)}, \tf1_micro: {round(val_f1_micro, 4)}, \tf1_weighted: {round(val_f1_weighted, 4)}, \tf1_binary: {round(val_f1_binary, 4)}')
        
        
        # This part uses loss to define the best model
        if best_loss is None:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_loss = val_loss
            print('New Best Model!')
            print(f'val_loss = {val_loss}; best_epoch = {best_epoch}')

        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_loss = val_loss
            print('New Best Model!')
            print(f'val_loss = {val_loss}; best_epoch = {best_epoch}')
            
            
        
        '''
        # This part PR AUC loss to define the best model
        if best_PR_AUC is None:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_PR_AUC = val_auc_precision_recall
            print('New Best Model!')
            print(f'val_auc_precision_recall = {val_auc_precision_recall}; best_epoch = {best_epoch}')

        if val_auc_precision_recall > best_PR_AUC:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_PR_AUC = val_auc_precision_recall
            print('New Best Model!')
            print(f'val_auc_precision_recall = {val_auc_precision_recall}; best_epoch = {best_epoch}')
        '''

        '''
        # This part uses brier score to define the best model
        if best_brier_score is None:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_brier_score = val_brier_score_loss
            print('New Best Model!')
            print(f'val_brier_score_loss = {val_brier_score_loss}; best_epoch = {best_epoch}')

        if val_brier_score_loss < best_brier_score:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_brier_score = val_brier_score_loss
            print('New Best Model!')
            print(f'val_brier_score_loss = {val_brier_score_loss}; best_epoch = {best_epoch}')
        '''

        print('')
   
    f1, precision, recall, accuracy, loss, roc_auc, auc_precision_recall, geometric_mean_score, brier_score_loss, f1_macro, f1_micro, f1_weighted, f1_binary = test(best_model, test_loader, device, loss_op)

    print(f'root path: {root_path}')
    print(f'best_epoch = {best_epoch}')
    print(f'Accuracy: {round(accuracy, 4)}, \tPrecision: {round(precision, 4)}, \tRecall: {round(recall, 4)}, \tF1: {round(f1, 4)}, \t\tROC AUC: {round(roc_auc, 4)}, \tPrecision-Recall AUC: {round(auc_precision_recall, 4)}')
    print(f'G-Means: {round(geometric_mean_score, 4)}, \tBrier Score: {round(brier_score_loss, 4)}, \tf1_macro: {round(f1_macro, 4)}, \tf1_micro: {round(f1_micro, 4)}, \tf1_weighted: {round(f1_weighted, 4)}, \tf1_binary: {round(f1_binary, 4)}')
    
    return [accuracy, precision, recall, f1, best_epoch, roc_auc, auc_precision_recall, geometric_mean_score, brier_score_loss, f1_macro, f1_micro, f1_weighted, f1_binary]



import numpy as np
if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    # Load options from a JSON file
    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    # Training models with diferent neftune_noise_alpha values
    for model_index in range(options["number_of_models"]):
        
        # Defining random seed
        torch.manual_seed(123)
        
        neftune_noise_alpha = options["initial_neftune_noise_alpha"] + model_index*options["neftune_noise_alpha_step_size"]

        print('\n\n_____________________________________________________________________________________________________________________________')
        print(f'Starting the training with neftune_noise_alpha = {neftune_noise_alpha}')
        # Run the run() function for multiple datasets and store results
        results = []
        index = 0
        for path in [
            'produced_data/datasets/dataset0',
            'produced_data/datasets/dataset1',
            'produced_data/datasets/dataset2',
            'produced_data/datasets/dataset3',
            'produced_data/datasets/dataset4',
            'produced_data/datasets/dataset5',
            'produced_data/datasets/dataset6',
            'produced_data/datasets/dataset7',
            'produced_data/datasets/dataset8',
            'produced_data/datasets/dataset9',
        ]:
            print('\n_____________________________________________________________________________________________________________________________')
            print(f'Dataset {index}')
            index += 1
            results.append(run(path, neftune_noise_alpha))

            partial_result_dataset_index = 0
            print('\nApproximate Results so far: ')
            for result in results:
                print(f'Dataset {partial_result_dataset_index} -> \tAccuracy: {round(result[0],4)}, \tPrecision: {round(result[1],4)}, \tRecall: {round(result[2],4)}, \tF1: {round(result[3],4)}, \tBest Epoch = {round(result[4],4)}, \tROC AUC = {round(result[5],4)}, \tAUC Precision Recall = {round(result[6],4)}')
                print(f'Dataset {partial_result_dataset_index} -> \tG-Mean = {round(result[7],4)}, \tBrier Score = {round(result[8],4)}, \tf1_macro: {round(result[9], 4)}, \tf1_micro: {round(result[10], 4)}, \tf1_weighted: {round(result[11], 4)}, \tf1_binary: {round(result[12], 4)}')
                print('')
                partial_result_dataset_index += 1

        print(f"\nFinal Approximate Results:")
        final_result_dataset_index = 0
        for result in results:
            print(f'Dataset {final_result_dataset_index} -> \tAccuracy: {round(result[0],4)}, \tPrecision: {round(result[1],4)}, \tRecall: {round(result[2],4)}, \tF1: {round(result[3],4)}, \tBest Epoch = {round(result[4],4)}, \tROC AUC = {round(result[5],4)}, \tAUC Precision Recall = {round(result[6],4)}')
            print(f'Dataset {final_result_dataset_index} -> \tG-Mean = {round(result[7],4)}, \tBrier Score = {round(result[8],4)}, \tf1_macro: {round(result[9], 4)}, \tf1_micro: {round(result[10], 4)}, \tf1_weighted: {round(result[11], 4)}, \tf1_binary: {round(result[12], 4)}')
            print('')            
            final_result_dataset_index += 1
        
        # Generate descriptive messages based on the loaded options
        # to inform about the type of experiment being conducted
        print()
        if options["user_embeddings"]:
            if options["retweet_embeddings"]:
                if options["embedder_type"].lower() == "glove":
                    print(f"The average metrics obtained using profile embeddings, using retweet embeddings, using GloVe as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
                elif options["embedder_type"].lower() == "bertweet":
                    print(f"The average metrics obtained using profile embeddings, using retweet embeddings, using BERTweet as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
            else:
                if options["embedder_type"].lower() == "glove":
                    print(f"The average metrics obtained using profile embeddings, not using retweet embeddings, using GloVe as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
                elif options["embedder_type"].lower() == "bertweet":
                    print(f"The average metrics obtained using profile embeddings, not using retweet embeddings, using BERTweet as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
        else:
            if options["retweet_embeddings"]:
                if options["embedder_type"].lower() == "glove":
                    print(f"The average metrics obtained not using profile embeddings, using retweet embeddings, using GloVe as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
                elif options["embedder_type"].lower() == "bertweet":
                    print(f"The average metrics obtained not using profile embeddings, using retweet embeddings, using BERTweet as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
            else:
                if options["embedder_type"].lower() == "glove":
                    print(f"The average metrics obtained not using profile embeddings, not using retweet embeddings, using GloVe as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")
                elif options["embedder_type"].lower() == "bertweet":
                    print(f"The average metrics obtained not using profile embeddings, not using retweet embeddings, using BERTweet as an embedder, and neftune_noise_alpha as {neftune_noise_alpha} are:")


        # Extract and calculate accuracies from the results
        accuracies = []        
        for result in results:
            accuracies.append(result[0])

        accuracies = np.array(accuracies)

        # Print statistics about accuracy and other metrics
        print(f'Mean accuracies: {accuracies.mean()} Std: {accuracies.std()}')

        # Calculate and print precision, recall, and F1-score statistics
        precisions = []        
        for result in results:
            precisions.append(result[1])

        if all([type(pr) is not str for pr in precisions]):
            precisions = np.array(precisions)
            print(f'Mean precisions: {precisions.mean()} Std: {precisions.std()}')
        else:
            print('Mean precision and std NaN')


        recalls = []        
        for result in results:
            recalls.append(result[2])

        if all([type(pr) is not str for pr in recalls]):
            recalls = np.array(recalls)
            print(f'Mean recalls: {recalls.mean()} Std: {recalls.std()}')
        else:
            print('Mean recall and std NaN')


        f1s = []        
        for result in results:
            f1s.append(result[3])

        if all([type(pr) is not str for pr in f1s]):
            f1s = np.array(f1s)
            print(f'Mean f1s: {f1s.mean()} Std: {f1s.std()}')
        else:
            print('Mean f1 and std NaN')


        best_epochs_in_strings = []        
        for result in results:
            best_epochs_in_strings.append(str(result[4]))

        best_epochs_in_integers = []
        for result in results:
            best_epochs_in_integers.append(result[4])

        ROC_AUCs = []        
        for result in results:
            ROC_AUCs.append(result[5])

        ROC_AUCs = np.array(ROC_AUCs)

        print(f'Mean ROC AUC: {ROC_AUCs.mean()} Std: {ROC_AUCs.std()}')


        AUC_precision_recalls = []        
        for result in results:
            AUC_precision_recalls.append(result[6])

        AUC_precision_recalls = np.array(AUC_precision_recalls)

        print(f'Mean AUC Precision Recall: {AUC_precision_recalls.mean()} Std: {AUC_precision_recalls.std()}')


        G_Means = []        
        for result in results:
            G_Means.append(result[7])

        G_Means = np.array(G_Means)

        print(f'Mean G-Means: {G_Means.mean()} Std: {G_Means.std()}')


        Brier_scores = []        
        for result in results:
            Brier_scores.append(result[8])

        Brier_scores = np.array(Brier_scores)

        print(f'Mean Brier Score: {Brier_scores.mean()} Std: {Brier_scores.std()}')


        f1_macros = []  
        f1_micros = []  
        f1_weighteds = []  
        f1_binarys = []  

        for result in results:
            f1_macros.append(result[9])
            f1_micros.append(result[10])
            f1_weighteds.append(result[11])
            f1_binarys.append(result[12])

        f1_macros = np.array(f1_macros)
        f1_micros = np.array(f1_micros)
        f1_weighteds = np.array(f1_weighteds)
        f1_binarys = np.array(f1_binarys)

        print(f'Mean f1_macros: {f1_macros.mean()} Std: {f1_macros.std()}')
        print(f'Mean f1_micros: {f1_micros.mean()} Std: {f1_micros.std()}')
        print(f'Mean f1_weighteds: {f1_weighteds.mean()} Std: {f1_weighteds.std()}')
        print(f'Mean f1_binarys: {f1_binarys.mean()} Std: {f1_binarys.std()}')
        

        # Saving the results in a csv file
        embedder = options["embedder_type"].lower()
        if options["user_embeddings"] == True:
            profiles = 1
        else: 
            profiles = 0
        if options["retweet_embeddings"] == True:
            retweets = 1
        else:
            retweets = 0

        Mean_accuracies = accuracies.mean()
        Mean_precisions = precisions.mean()
        Mean_recalls = recalls.mean()
        Mean_f1s = f1s.mean()
        Mean_ROC_AUC = ROC_AUCs.mean()
        Mean_AUC_precision_recall = AUC_precision_recalls.mean()
        Mean_G_Means = G_Means.mean()
        Mean_brier_score = Brier_scores.mean()
        Mean_f1_macro = f1_macros.mean()
        Mean_f1_micro = f1_micros.mean()
        Mean_f1_weighted = f1_weighteds.mean()
        Mean_f1_binary = f1_binarys.mean()

        Std_accuracies = accuracies.std()
        Std_precisions = precisions.std()
        Std_recalls = recalls.std()
        Std_f1s = f1s.std()
        Std_ROC_AUC = ROC_AUCs.std()
        Std_AUC_precision_recall = AUC_precision_recalls.std()
        Std_G_Means = G_Means.std()
        Std_brier_score = Brier_scores.std()
        Std_f1_macro = f1_macros.std()
        Std_f1_micro = f1_micros.std()
        Std_f1_weighted = f1_weighteds.std()
        Std_f1_binary = f1_binarys.std()

        Best_epochs_string  = ' - '.join(best_epochs_in_strings)
        Mean_best_epoch = np.mean(best_epochs_in_integers)

        
        # Creating a DataFrame with the main results
        df = pd.DataFrame({
            'Embedder': [embedder],
            'Profiles': [profiles],
            'Retweets': [retweets],
            'neftune_noise_alpha': [neftune_noise_alpha],
            'Mean_accuracies': [Mean_accuracies],
            'Mean_precisions': [Mean_precisions],
            'Mean_recalls': [Mean_recalls],
            'Mean_f1s': [Mean_f1s],
            'Mean_ROC_AUC': [Mean_ROC_AUC],
            'Mean_AUC_precision_recall': [Mean_AUC_precision_recall],
            'Mean_G_Means': [Mean_G_Means],
            'Mean_brier_score': [Mean_brier_score],
            'Mean_f1_macro': [Mean_f1_macro],
            'Mean_f1_micro': [Mean_f1_micro],
            'Mean_f1_weighted': [Mean_f1_weighted],
            'Mean_f1_binary': [Mean_f1_binary],
            'Std_accuracies': [Std_accuracies],
            'Std_precisions': [Std_precisions],
            'Std_recalls': [Std_recalls],
            'Std_f1s': [Std_f1s],
            'Std_ROC_AUC': [Std_ROC_AUC],
            'Std_AUC_precision_recall': [Std_AUC_precision_recall],
            'Std_G_Means': [Std_G_Means],
            'Std_brier_score': [Std_brier_score],
            'Std_f1_macro': [Std_f1_macro],
            'Std_f1_micro': [Std_f1_micro],
            'Std_f1_weighted': [Std_f1_weighted],
            'Std_f1_binary': [Std_f1_binary],
            'Best_epochs': [Best_epochs_string],
            'Mean best epoch': [Mean_best_epoch]
        })

        # Saving the DataFrame in a csv file
        # Checking if the output file already exists and if it's empty
        summarized_output_file = 'summarized_output.csv'
        if os.path.exists(summarized_output_file) and not pd.read_csv(summarized_output_file).empty:
            # If the file exists and is not empty, append the new results
            df.to_csv(summarized_output_file, mode='a', header=False, index=False)
        else:
            # If the file doesn't exist or is empty, write the new results with headers
            df.to_csv(summarized_output_file, index=False)

        # Creating the complete output
        kfold_index = 1
        for result in results:
            k_fold_df = pd.DataFrame({
                'k-fold': [kfold_index],
                'Embedder': [embedder],
                'Profiles': [profiles],
                'Retweets': [retweets],
                'neftune_noise_alpha': [neftune_noise_alpha],
                'accuracy': [result[0]],
                'precision': [result[1]],
                'recall': [result[2]],
                'f1': [result[3]],
                'ROC_AUC': [result[5]],
                'AUC_precision_recall': [result[6]],
                'G_Mean': [result[7]],
                'brier_score': [result[8]],
                'f1_macro': [result[9]],
                'f1_micro': [result[10]],
                'f1_weighted': [result[11]],
                'f1_binary': [result[12]],
                'best_epoch': [result[4]]
                })
            
            kfold_index += 1

            # Saving the DataFrame in a csv file
            # Checking if the output file already exists and if it's empty
            complete_output_file = 'complete_output.csv'
            if os.path.exists(complete_output_file) and not pd.read_csv(complete_output_file).empty:
                # If the file exists and is not empty, append the new results
                k_fold_df.to_csv(complete_output_file, mode='a', header=False, index=False)
            else:
                # If the file doesn't exist or is empty, write the new results with headers
                k_fold_df.to_csv(complete_output_file, index=False)
