#!/usr/bin/env python

#
# Compute retweet embeddings.
#

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
        embeddings_file,
        users_embeddings_lookup,
        not_in_lookup_embedding
    ): #Acho que já está certa essa função
        self._user_profiles_path = user_profiles_path
        self._retweets_embeddings_path = retweets_embeddings_path
        self._embeddings_file = embeddings_file
        self._users_embeddings_lookup = users_embeddings_lookup
        self._not_in_lookup_embedding = not_in_lookup_embedding


    def _strip_retweet(self, retweet, embedder): # Acho que já está certa essa função
        if 'done' in retweet and retweet['done'] !=  'OK':
            text = ''
            #retweet_author = retweet['id']
            # Caso não tenha o id do usuário
            try:
                retweet_author = retweet['id']
            except KeyError:
                return None
            
            #retweet = models.Tweet(int(retweet['status']['id']))
            # Caso não exista um retweet
            try:
                retweet = models.Tweet(int(retweet['status']['id']))
            except KeyError:
                return None

        else:
            #text = retweet['status']['text']
            try:
                text = retweet['status']['text']
            except KeyError:
                text = None
            #retweet_author = retweet['id']
            try:
                retweet_author = retweet['id']
            except KeyError:
                return None

            # Caso não exista um retweet
            try:
                retweet = models.Tweet(int(retweet['status']['id']))
            except KeyError:
                return None



        retweet.text = text
        retweet.user = retweet_author
        

        retweet_id_and_embedding = {}
        retweet_id_and_embedding['rewteet_id'] = retweet.id
        retweet_id_and_embedding['user'] = retweet.user
        graphsage_embedding = self._users_embeddings_lookup.get(str(retweet_id_and_embedding['rewteet_id']), None)
        if graphsage_embedding is None:
            graphsage_embedding = self._not_in_lookup_embedding.tolist()
        retweet_id_and_embedding["embedding"] = embedder.embed(retweet).tolist() + graphsage_embedding
        return retweet_id_and_embedding


    def run(self): # Acho que já está certa essa função
                   # Ver de no futuro criar os embeddings dos usuarios e dos retweets ao mesmo tempo para só abrir cada arquivo 1 vez
        # Create output dir
        logging.info("Will output user embeddings to {}".format(self._retweets_embeddings_path))
        os.makedirs(self._retweets_embeddings_path, exist_ok=True)

        bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base")
        embedder = embeddings.RetweetContentEmbedder(bertweet_model=bertweet_model)

        length = len(list(os.scandir(self._user_profiles_path))) # Retweets e user_profiles estão salvos em um mesmo json?
        for fentry in tqdm(os.scandir(self._user_profiles_path), total=length):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    retweet = json.load(json_file)
                    retweet_id_and_embedding = self._strip_retweet(retweet, embedder)
                    if retweet_id_and_embedding is not None:

                        outfile = "{}/{}.json".format(self._retweets_embeddings_path, retweet_id_and_embedding['user'])
                        with open(outfile, "w") as out_json_file:
                            logging.debug("Writing user embeddings to file {}".format(outfile))
                            json.dump(retweet_id_and_embedding, out_json_file)


def run(args): # Acho que já está certa essa função

    logging.info("Loading dataset")

    user_profiles_path = "{}/user_profiles".format(args.input_dir)
    retweets_embeddings_path = "{}/retweets_embeddings".format(args.dataset_root)

    logging.info("Loading retweet embeddings using BERTweet")
    with open(os.path.join(args.dataset_root, "datasets", "dataset0", "users_graphsage_embeddings_lookup.json")) as f:
        users_embeddings_lookup = json.load(f)

    dataset = UserRetweets(
        user_profiles_path=user_profiles_path,
        retweets_embeddings_path=retweets_embeddings_path,
        embeddings_file=args.embeddings_file,
        users_embeddings_lookup=users_embeddings_lookup,
        not_in_lookup_embedding=np.zeros(len(list(users_embeddings_lookup.values())[0]))
    )

    dataset.run()


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