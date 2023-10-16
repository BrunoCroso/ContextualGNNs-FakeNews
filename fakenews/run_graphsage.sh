#!/usr/bin/env bash

python train_graphsage.py --dataset-root produced_data/dataset0 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset1 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset2 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset3 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset4 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset5 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset6 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset7 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset8 --epochs 10
python train_graphsage.py --dataset-root produced_data/dataset9 --epochs 10

echo Please upload the files...
