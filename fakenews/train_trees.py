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
            dims = torch.tensor(output.size(1))
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
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


    #print('tn:%s fp:%s fn:%s tp:%s' % (tn, fp, fn, tp))
    return f1, precision, recall, accuracy, final_loss

from sklearn.utils import class_weight

def run(root_path):
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

        #number_of_samples = min(len(dataset_real), len(dataset_fake))
        #number_of_samples = 5000000                #COMENTEI ESTA LINHA PARA TESTAR

        #number_of_real = len(dataset_real)
        #number_of_fake = len(dataset_fake)
        #if i == 0:
        #    multiply_by = number_of_real / number_of_fake
        #    print('multiply by')
        #    print(multiply_by)
        #    multiply_by = int(multiply_by)
        #    dataset_fake = dataset_fake * multiply_by

        #This part was used to balance the datasets -> it was moved below ___________________DELETE
        
        number_of_real = len(dataset_real)
        number_of_fake = len(dataset_fake)

        multiply_by = int(number_of_real / number_of_fake)
        dataset_fake = dataset_fake * multiply_by

        number_of_samples = min(len(dataset_real), len(dataset_fake)) #TESTE
        dataset = dataset_real[:number_of_samples] + dataset_fake[:number_of_samples]
        #print('number of samples')
        #print(i, len(dataset))

        datasets.append(dataset)

    train_labels = [i.y.item() for i in datasets[0]]
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                        y=train_labels)
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
    model.batch_norm.neftune_noise_alpha = options["neftune_noise_alpha"]
    model.batch_norm.register_forward_hook(neftune_post_forward_hook)
    loss_op = torch.nn.NLLLoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    # Creating variables to save the best model
    best_model = None
    best_loss = None
    best_epoch = 1

    # Start training
    for epoch in range(1, 61):
        logging.info("Starting epoch {}".format(epoch))
        loss = train(model, train_loader, device, optimizer, loss_op)
        print(f'Loss: {loss}')

        # Testing the model in the validation set
        val_f1, val_precision, val_recall, val_accuracy, val_loss = test(model, val_loader, device, loss_op)
        if best_loss is None:
            best_loss = val_loss

        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_loss = val_loss
            print('New Best Model!')
            print(f'val_loss = {val_loss}; best_epoch = {best_epoch}')

   
    f1, precision, recall, accuracy, loss = test(best_model, test_loader, device, loss_op)

    print(f'root path: {root_path}')
    print(f'best_epoch = {best_epoch}')
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} F1: {f1}')
    return [accuracy, precision, recall, f1, best_epoch]



import numpy as np
if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    # Defining random seed
    torch.manual_seed(123)


    # Run the 'run' function for multiple datasets and store results
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
        results.append(run(path))

    print(f"Results = \n {results}") #DELETE____________________________________________
    
    # Load options from a JSON file
    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    neftune_noise_alpha = options["neftune_noise_alpha"]
    
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
    print(f'Mean accuracies {accuracies.mean()} Std: {accuracies.std()}')

    # Calculate and print precision, recall, and F1-score statistics
    precisions = []        
    for result in results:
        precisions.append(result[1])

    if all([type(pr) is not str for pr in precisions]):
        precisions = np.array(precisions)
        print(f'Mean precisions {precisions.mean()} Std: {precisions.std()}')
    else:
        print('Mean precision and std NaN')


    recalls = []        
    for result in results:
        recalls.append(result[2])

    if all([type(pr) is not str for pr in recalls]):
        recalls = np.array(recalls)
        print(f'Mean recalls {recalls.mean()} Std: {recalls.std()}')
    else:
        print('Mean recall and std NaN')


    f1s = []        
    for result in results:
        f1s.append(result[3])

    if all([type(pr) is not str for pr in f1s]):
        f1s = np.array(f1s)
        print(f'Mean f1s {f1s.mean()} Std: {f1s.std()}')
    else:
        print('Mean f1 and std NaN')


    best_epochs = []        
    for result in results:
        best_epochs.append(result[4])


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
    neftune_noise_alpha = options["neftune_noise_alpha"]

    Mean_accuracies = accuracies.mean()
    Mean_precisions = precisions.mean()
    Mean_recalls = recalls.mean()
    Mean_f1s = f1s.mean()
    Std_accuracies = accuracies.std()
    Std_precisions = precisions.std()
    Std_recalls = recalls.std()
    Std_f1s = f1s.std()
    Best_epochs_string  = '; '.join(best_epochs)

    
    # Criating a DataFrame with the main results
    df = pd.DataFrame({
        'Embedder': [embedder],
        'Profiles': [profiles],
        'Retweets': [retweets],
        'neftune_noise_alpha': [neftune_noise_alpha],
        'Mean_accuracies': [Mean_accuracies],
        'Mean_precisions': [Mean_precisions],
        'Mean_recalls': [Mean_recalls],
        'Mean_f1s': [Mean_f1s],
        'Std_accuracies': [Std_accuracies],
        'Std_precisions': [Std_precisions],
        'Std_recalls': [Std_recalls],
        'Std_f1s': [Std_f1s],
        'Best_epochs': [Best_epochs_string]
    })

    # Saving the DataFrame in a csv file
    # Checking if the output file already exists and if it's empty
    output_file = 'output.csv'
    if os.path.exists(output_file) and not pd.read_csv(output_file).empty:
        # If the file exists and is not empty, append the new results
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # If the file doesn't exist or is empty, write the new results with headers
        df.to_csv(output_file, index=False)

