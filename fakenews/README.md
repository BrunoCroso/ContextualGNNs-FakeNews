# Astroturfing project - FakeNews dataset


Unzip all data into folder `../raw_data`. It should contain the following directory 
structure. 

```
politifact
user_followers
user_profiles
```

# Compute user embeddings

Download user embeddings in the root folder of the problem and unzip in `../raw_data`. 

```
curl -LO https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```

Compute using 

```
./compute_user_embeddings.py --input-dir=../raw_data --output-dir=../raw_data --embeddings-file=../raw_data/glove.twitter.27B.100d.txt
```

you should get a directory `../raw_data/user_embeddings` containing one json file per user. 

## Compute user labels 

Compute using 

```
./compute_user_labels.py --input-dir=../raw_data --output-dir=../raw_data
```

you should get a directory `../raw_data/user_labels` containing one json file per user. 

## Compute trees

First you need to preprocess the dataset using 

```
./dataset_preprocess.py
```

This will create a folder `tweets1`. Then run 

```
./create_trees.py
```

which will create a folder `trees2`.


## Train a Graph Neural Network

* Download glove embeddings and extract the zip file. 
  Available at https://nlp.stanford.edu/projects/glove/ (recommended file: "Twitter (2B tweets): glove.twitter.27B.zip")
* Run `compute_user_label.py` script file
* Run `user_to_graph.py` script file (e.g. `python users_to_graph.py --input-dir ../raw_data --embeddings-file ../raw_data/glove.twitter.27B/glove.twitter.27B.200d.txt`)
* Run `train_graphsage.py` script file (e.g. `python train_graphsage.py --user-labels-dir ../raw_data/user_labels`)


# How to run the whole process
Step 1.
Connect to the hua server (10.100.54.29) and execute the following commands:
```
cd astroturfing/fakenews
./run.sh
```

Step 2. When the above script pauses its execution, run the following commands locally:
```
./download_graphsage_data.sh
./run_graphsage.sh
./upload_graphsage_produce_data.sh
```

Step 3. When the above scripts finish their execution, go to the hua server and continue the execution of the script.

Step 4. When the script of the hua server finishes its execution, run the following commands locally:
```
./download_gat_data.sh
./run_gat.sh
```