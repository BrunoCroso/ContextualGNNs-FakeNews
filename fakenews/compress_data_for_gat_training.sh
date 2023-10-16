#!/usr/bin/env bash

cd produced_data

for i in {0..9}
do
  rm -r gat_dataset$i
  mkdir gat_dataset$i
done

echo Copying train, test, val trees
for i in {0..9}
do
  cp -r datasets/dataset$i/train gat_dataset$i/train
  cp -r datasets/dataset$i/test gat_dataset$i/test
  cp -r datasets/dataset$i/val gat_dataset$i/val
done

echo Compressing datasets for GAT training
for i in {0..9}
do
  tar -czf gat_dataset$i.tar.gz gat_dataset$i
done
