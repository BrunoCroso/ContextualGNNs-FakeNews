# Organizing the data


After copying the repository,, unzip all data into folder `../raw_data`. It should contain the following directory 
structure. 

```
politifact
user_followers
user_profiles
```

# Downloading GloVe

Download user embeddings in the root folder of the problem and unzip in `../raw_data`. 

```
curl -LO https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```


# Downloading the necessary libraries

Download the necessary packages indicated in the requirements.txt file

# How to run the whole process

Define the model’s characteristics using the options.json file

Use the run.sh file to run the entire process