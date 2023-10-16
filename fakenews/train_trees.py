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

from tqdm import tqdm


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
        for key in ["delay", "protected", "following_count", "listed_count", "statuses_count", "followers_count", "favourites_count", "verified",]:
            as_list.append(float(node[key]))
        as_list.extend(node["user_embedding"])
        as_list.extend(node["retweet_embedding"])
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
    return label, number_of_features, result


from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.norm import GraphSizeNorm
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.batch_norm = BatchNorm(num_features)
        self.conv1 = GATConv(num_features, 32, heads=4, dropout=0.5)
        self.conv2 = GATConv(32 * 4, num_classes, heads=1, concat=False, dropout=0.5)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.batch_norm(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)


def train(model, loader, device, optimizer, loss_op):
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
    print(len(loader.dataset))
    return total_loss / len(loader.dataset)

from sklearn.metrics import confusion_matrix

@torch.no_grad()
def test(model, loader, device):
    model.eval()

    ys, preds = [], []
    for data in loader:
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

    #print('y:##############')
    #print(y)
    #print('pred###############')
    #print(pred)


    #print('tn:%s fp:%s fn:%s tp:%s' % (tn, fp, fn, tp))
    return f1, precision, recall, accuracy

from sklearn.utils import class_weight

def run(root_path):

    logging.info("Loading dags dataset")

    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'val')
    test_path = os.path.join(root_path, 'test')

    datasets = []
    for i, path in enumerate([train_path, val_path, test_path]):
        number_of_reals = 0
        dataset_fake = []
        dataset_real = []
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                label, number_of_features, t = tree_to_data(fentry.path)
                if label == "real":
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
        #print('number of samples')
        #print(i, len(dataset))
        datasets.append(dataset)

    # 0 = true, 1 = fake
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
    #weights = [4.0, 0.5]
    #print('weights!!')
    #print(weights)
    #class_weights = torch.FloatTensor(weights).cuda()
    #self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    #loss_op = torch.nn.NLLLoss(weight=class_weights)
    loss_op = torch.nn.NLLLoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    # Start training
    for epoch in range(1, 21):
        logging.info("Starting epoch {}".format(epoch))
        loss = train(model, train_loader, device, optimizer, loss_op)
        print(loss)
        #if epoch % 5 == 0:
        #    val_f1, val_precision, val_recall, val_accuracy = test(model, val_loader, device)
        #    #print('Epoch: {:02d}, Loss: {:.4f}, Val F1: {:.4f} Val Prec: {:.4f} Val Rec: {:.4f} Val Acc: {:.4f}'.format(epoch, loss, val_f1, val_precision, val_recall, val_accuracy))

    f1, precision, recall, accuracy = test(model, test_loader, device)
    print(root_path)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall} F1: {f1}')
    return [accuracy, precision, recall, f1]
    


import numpy as np
if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )


    results = []
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
        results.append(run(path))
    accuracies = [res[0] for res in results]
    accuracies = np.array(accuracies)
    print()
    print(f'Mean accuracy {accuracies.mean()} Std: {accuracies.std()}')

    precisions = [res[1] for res in results]
    if all([type(pr) is not str for pr in precisions]):
        precisions = np.array(precisions)
        print(f'Mean precision {precisions.mean()} Std: {precisions.std()}')
    else:
        print('Mean precision and std NaN')

    recalls = [res[2] for res in results]
    if all([type(pr) is not str for pr in recalls]):
        recalls = np.array(recalls)
        print(f'Mean recall {recalls.mean()} Std: {recalls.std()}')
    else:
        print('Mean recall and std NaN')

    f1s = [res[3] for res in results]
    if all([type(pr) is not str for pr in f1s]):
        f1s = np.array(f1s)
        print(f'Mean f1 {f1s.mean()} Std: {f1s.std()}')
    else:
        print('Mean f1 and std NaN')
