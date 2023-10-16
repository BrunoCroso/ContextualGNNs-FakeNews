#!/usr/bin/env bash

#rm -r produced_data/*
#rm -r tweets1
#rm -r trees2

#echo "##### RUN dataset_preprocess #####"
#python dataset_preprocess.py --ignore-dataset-pkl --sample-probability 0.1
#echo "##### RUN create_trees #####"
#python create_trees.py --tweets tweets1
#mv trees2 produced_data/trees
#echo "##### RUN generate_kfolds #####"
#python generate_kfolds.py --k 10 --val-size 0.5 #MUDEI AQUI PARA TESTAR
#echo "##### RUN compute_user_labels #####"
#python compute_user_labels.py --input-dir=../raw_data
#echo "##### RUN users_to_graph #####"
#for i in {0..9} #MUDEI AQUI PARA TESTAR
#do
#  echo "## dataset$i ##"
#  python users_to_graph.py --input-dir ../raw_data --embeddings-file ../raw_data/glove.twitter.27B.100d.txt --dataset-root produced_data/datasets/dataset$i
#done


    # echo Compressing data for local graphsage training...

    # ./compress_data_for_graphsage_training.sh

    # read -p "Download the data locally, run the graphsage training, upload the results and then press [Enter] key to continue..."

#echo "##### RUN train_graphsage#####"
#for i in {0..9} #MUDEI AQUI PARA TESTAR
#do
#  python train_graphsage.py --dataset-root produced_data/datasets/dataset$i --epochs 10
#done

#echo "##### RUN compute_user_embeddings #####"
#for i in {0..1} #MUDEI AQUI PARA TESTAR
#do
#  echo "## dataset$i ##"
#  python compute_user_embeddings.py --input-dir ../raw_data --dataset-root produced_data/datasets/dataset$i --embeddings-file ../raw_data/glove.twitter.27B.100d.txt
#done

#echo "##### TEST RUN compute_user_embeddings #####"
#python compute_user_embeddings.py --input-dir ../raw_data --dataset-root produced_data --embeddings-file ../raw_data/glove.twitter.27B.100d.txt

#echo "##### RUN compute_retweet_embeddings #####"
#for i in {0..1} #MUDEI AQUI PARA TESTAR
#do
#  echo "## dataset$i ##"
#  python compute_retweet_embeddings.py --input-dir ../raw_data --dataset-root produced_data/datasets/dataset$i --embeddings-file ../raw_data/glove.twitter.27B.100d.txt
#done


#echo "##### TEST RUN compute_retweet_embeddings #####"
#python compute_retweet_embeddings.py --input-dir ../raw_data --dataset-root produced_data --embeddings-file ../raw_data/glove.twitter.27B.100d.txt


#echo "##### RUN add_trees_information #####"
#for i in {0..1} #MUDEI AQUI PARA TESTAR
#do
#  echo "## dataset$i ##"
#  python add_trees_information.py --num-datasets 2 --dataset-root produced_data #/datasets/dataset$i
#done

echo "##### TEST RUN add_trees_information #####"
python add_trees_information.py --num-datasets 10 --dataset-root produced_data #/datasets/dataset$i



    # echo Compressing data for local GAT training...

    # ./compress_data_for_gat_training.sh

    # echo Download again the data locally. You can now run the final training of the model on your computer...

echo "##### RUN train_trees #####"

python train_trees.py --dataset-root produced_data/datasets/dataset0

echo "##### Process Finished #####"