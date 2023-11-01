#!/usr/bin/env python

#
# Compute user profile embeddings.
#

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


class UserProfiles: 
    def __init__(
        self,
        user_profiles_path,
        user_embeddings_path,
        embeddings_file
    ):
        self._user_profiles_path = user_profiles_path
        self._user_embeddings_path = user_embeddings_path
        self._embeddings_file = embeddings_file


    def _strip_user_profile(self, user_profile, user_embedder):
        '''
        Extract user information and generate embeddings from a user profile.
        '''

        if 'done' in user_profile and user_profile['done'] !=  'OK':
            description = ''
            user_profile = models.User(int(user_profile['user_id']))
        else:
            description = user_profile['description']
            user_profile = models.User(user_profile['id'])
        user_profile.description = description

        user = {}
        user['id'] = user_profile.id
        user["embedding"] = user_embedder.embed(user_profile).tolist()
        return user


    def run(self):
        '''  
        This method loads user profiles, computes embeddings for each user profile,
        and saves the results to JSON files in the specified output directory. The
        specific embedding method used (GloVe or BERTweet) depends on the options
        defined in "options.json" configuration file.
        '''

        # Create output dir
        logging.info("Will output user embeddings to {}".format(self._user_embeddings_path))
        os.makedirs(self._user_embeddings_path, exist_ok=True)

        with open("options.json", "r") as json_file:
            options = json.load(json_file)

        if options["embedder_type"].lower() == "glove":
            glove_embeddings = utils.load_glove_embeddings(self._embeddings_file)
            user_embedder = embeddings.GloVeUserEmbedder(glove_embeddings=glove_embeddings)        
        
        
        elif options["embedder_type"].lower() == "bertweet":
            bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base")
            user_embedder = embeddings.BERTweetUserEmbedder(bertweet_model=bertweet_model)


        length = len(list(os.scandir(self._user_profiles_path)))
        for fentry in tqdm(os.scandir(self._user_profiles_path), total=length):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    user_profile = json.load(json_file)
                    user = self._strip_user_profile(user_profile, user_embedder)

                    outfile = "{}/{}.json".format(self._user_embeddings_path, user['id'])
                    with open(outfile, "w") as out_json_file:
                        logging.debug("Writing user embeddings to file {}".format(outfile))
                        json.dump(user, out_json_file)


def run(args):
    '''
    This function reads the configuration options from "options.json" and determines whether
    user embeddings should be generated. If enabled, it selects the embedding method (GloVe or
    BERTweet) based on the options and initiates the user embeddings generation process.
    '''
    
    with open("options.json", "r") as json_file:
        options = json.load(json_file)

    if options["user_embeddings"] == True:

        if options["embedder_type"].lower() == "glove":
            print("\nEmbeddings will be generated using GloVe, as defined in options.json\n")
            user_embeddings_path = "{}/glove_user_embeddings".format(args.dataset_root)
            
        elif options["embedder_type"].lower() == "bertweet":
            print("\nEmbeddings will be generated using BERTweet, as defined in options.json\n")
            user_embeddings_path = "{}/bertweet_user_embeddings".format(args.dataset_root)
        

        user_profiles_path = "{}/user_profiles".format(args.input_dir)

        logging.info("Loading dataset")

        dataset = UserProfiles(
            user_profiles_path=user_profiles_path,
            user_embeddings_path=user_embeddings_path,
            embeddings_file=args.embeddings_file
        )

        dataset.run()

    else:
        print("\nEmbeddings will NOT be generated for user profiles, as defined in options.json\n")

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python compute_user_embeddings.py"
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
