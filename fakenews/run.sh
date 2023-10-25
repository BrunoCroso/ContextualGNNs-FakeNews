#!/usr/bin/env bash

#rm -r produced_data/*
rm -r tweets1
rm -r trees2

echo "##### Model informations #####" # Documentado
python infos.py
echo "##### RUN dataset_preprocess #####" # Documentado
python dataset_preprocess.py --ignore-dataset-pkl --sample-probability 1 # Devolver para 1
echo "##### RUN create_trees #####" 
python create_trees.py --tweets tweets1
mv trees2 produced_data/trees
echo "##### RUN generate_kfolds #####"
python generate_kfolds.py --k 10 --val-size 0.1 #MUDEI AQUI PARA TESTAR
echo "##### RUN compute_user_labels #####"
python compute_user_labels.py --input-dir=../raw_data


#echo "##### TEST RUN compute_user_embeddings #####"
#python compute_user_embeddings.py --input-dir ../raw_data --dataset-root produced_data --embeddings-file ../raw_data/glove.twitter.27B.100d.txt


#echo "##### TEST RUN compute_retweet_embeddings #####"
#python compute_retweet_embeddings.py --input-dir ../raw_data --dataset-root produced_data --embeddings-file ../raw_data/glove.twitter.27B.100d.txt


echo "##### TEST RUN add_trees_information #####"
python add_trees_information.py --num-datasets 10 --dataset-root produced_data



    # echo Compressing data for local GAT training...

    # ./compress_data_for_gat_training.sh

    # echo Download again the data locally. You can now run the final training of the model on your computer...

echo "##### RUN train_trees #####"
python train_trees.py --dataset-root produced_data/datasets/dataset0

echo "##### Process Finished #####"