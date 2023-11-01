#!/usr/bin/env python

'''
Compute retweet embeddings.
'''

# AINDA ESTÁ COMPLETO 

import argparse
import json
import os
import logging

import models 
import embeddings
import utils
import numpy as np

from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

class UserRetweets: 
    def __init__(
        self,
        user_profiles_path,
        retweets_embeddings_path,
        embeddings_file
    ):
        self._user_profiles_path = user_profiles_path
        self._retweets_embeddings_path = retweets_embeddings_path
        self._embeddings_file = embeddings_file


    def _strip_retweet(self, retweet, embedder):
        '''
        Extract and preprocess retweet information and generate retweet embeddings.
        '''

        if 'done' in retweet and retweet['done'] !=  'OK':
            text = ''
            try:
                retweet_author = retweet['id']
            except KeyError:
                return None
            
            try:
                retweet = models.Tweet(int(retweet['status']['id']))
            except KeyError:
                return None

        else:
            try:
                text = retweet['status']['text']
            except KeyError:
                text = None
            try:
                retweet_author = retweet['id']
            except KeyError:
                return None

            try:
                retweet = models.Tweet(int(retweet['status']['id']))
            except KeyError:
                return None

        retweet.text = text
        retweet.user = retweet_author
        
        retweet_id_and_embedding = {}
        retweet_id_and_embedding['rewteet_id'] = retweet.id
        retweet_id_and_embedding['user'] = retweet.user
        retweet_id_and_embedding["embedding"] = embedder.embed(retweet).tolist()
        return retweet_id_and_embedding


    def run(self): 
        '''
        This function iterates through retweet data, preprocesses retweets, and generates
        embeddings. The resulting retweet embeddings are saved to the specified output directory.
        '''

        # Create output dir
        logging.info("Will output user embeddings to {}".format(self._retweets_embeddings_path))
        os.makedirs(self._retweets_embeddings_path, exist_ok=True)

        with open("options.json", "r") as json_file:
            options = json.load(json_file)

        if options["embedder_type"].lower() == "glove":
            glove_embeddings = utils.load_glove_embeddings(self._embeddings_file)
            retweet_embedder = embeddings.GloVeRetweetContentEmbedder(glove_embeddings=glove_embeddings)        
        
        
        elif options["embedder_type"].lower() == "bertweet":
            bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base")
            retweet_embedder = embeddings.BERTweetRetweetContentEmbedder(bertweet_model=bertweet_model)


        length = len(list(os.scandir(self._user_profiles_path))) # Retweets e user_profiles estão salvos em um mesmo json?
        for fentry in tqdm(os.scandir(self._user_profiles_path), total=length):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    retweet = json.load(json_file)
                    retweet_id_and_embedding = self._strip_retweet(retweet, retweet_embedder)
                    if retweet_id_and_embedding is not None:

                        outfile = "{}/{}.json".format(self._retweets_embeddings_path, retweet_id_and_embedding['user'])
                        with open(outfile, "w") as out_json_file:
                            logging.debug("Writing user embeddings to file {}".format(outfile))
                            json.dump(retweet_id_and_embedding, out_json_file)


def run(args): 
    '''
    This function first loads the options from the "options.json" file. Depending on the selected
    embedding type (GloVe or BERTweet) and the retweet embedding setting in the options, it generates
    retweet embeddings and saves them to the specified output directory. 
    '''
    
    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    if options["retweet_embeddings"] == True:

        if options["embedder_type"].lower() == "glove":
            print("\nEmbeddings will be generated using GloVe, as defined in options.json\n")
            retweets_embeddings_path = "{}/glove_retweets_embeddings".format(args.dataset_root)
                
        elif options["embedder_type"].lower() == "bertweet":
            print("\nEmbeddings will be generated using BERTweet, as defined in options.json\n")
            retweets_embeddings_path = "{}/bertweet_retweets_embeddings".format(args.dataset_root)
    
        user_profiles_path = "{}/user_profiles".format(args.input_dir)

        logging.info("Loading dataset")

        dataset = UserRetweets(
            user_profiles_path=user_profiles_path,
            retweets_embeddings_path=retweets_embeddings_path,
            embeddings_file=args.embeddings_file
        )

        dataset.run()

    else:
        print("\nEmbeddings will NOT be generated for retweet content, as defined in options.json\n")


if __name__ == "__main__": # As pastas e o resto do conteúdo são passados no terminal )no run.sh)

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python compute_retweet_embeddings.py"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing the fakenewsnet dataset",
        dest="input_dir",
        type=str, 
        required=True
    )
    parser.add_argument(
        "--dataset-root",
        help="Output directory to export",
        dest="dataset_root",
        type=str,
        required=True
    )
    parser.add_argument(
        "--embeddings-file",
        help="Embeddings filepath",
        dest="embeddings_file",
        type=str,
        required=True
    )    
    args = parser.parse_args()
    run(args)