import os
import json
import pickle
import random
import numpy as np
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd

stop_words = set(stopwords.words('english'))

def collect_user_ids(dirpath):
    """

    :param dirpath:
    :return:
    """
    user_ids = set()
    for fentry in os.scandir(dirpath):
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                full_tweet = json.load(json_file)
                user_ids.add(full_tweet["user"]["id_str"])
                retweeted_status = full_tweet.get("retweeted_status")
                if retweeted_status is not None:
                    user_ids.add(retweeted_status["user"]["id_str"])
    return list(user_ids)


def write_object_to_pickle_file(path, obj):
    """

    :param path:
    :param obj:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle_from_file(path):
    """

    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_glove_embeddings(filepath: str) -> Dict:
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index

def generate_tokens_from_text(text:str, lowercased:bool=True, stopwords_removed:bool=True) -> List[str]:
    """Convert a text to a list of tokens.
    """
    if lowercased:
        text = text.lower()
    tokens = word_tokenize(text)
    if stopwords_removed:
        tokens = [w for w in tokens if not w in stop_words]
    return tokens

def create_user_labels_df(user_ids, user_labels_dir):
    labels = []
    indexes = []
    count1 = 0
    count2 = 0
    for user_id in tqdm(user_ids):
        if not os.path.exists('%s/%s.json' % (user_labels_dir, user_id)):
            count1 += 1
            indexes.append(int(user_id))
            labels.append(0)
            continue

        with open('%s/%s.json' % (user_labels_dir, user_id)) as json_file:
            user_label = json.load(json_file)
            indexes.append(int(user_label['id']))
            if user_label['fake'] >= user_label['real']/4.0:
                labels.append(1)
            else:
                labels.append(0)
            count2 += 1
    df = pd.DataFrame(labels, index=indexes, columns=['label'])
    print('We set zero label to missing %s users' % (count1))
    print('We set correct labels to %s users' % (count2))
    return df


def create_json_files_iterator(path, sample_probability=None):
    allfiles = []
    for fentry in os.scandir(path):
        if fentry.path.endswith(".json") and fentry.is_file():
            if sample_probability is not None: 
                if random.uniform(0, 1) > sample_probability:
                    continue
            allfiles.append(fentry.path)
    return allfiles
