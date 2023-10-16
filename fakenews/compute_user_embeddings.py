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

from tqdm import tqdm


class UserProfiles: 
    def __init__(
        self,
        user_profiles_path,
        user_embeddings_path,
        embeddings_file,
        users_embeddings_lookup,
        not_in_lookup_embedding
    ):
        self._user_profiles_path = user_profiles_path
        self._user_embeddings_path = user_embeddings_path
        self._embeddings_file = embeddings_file
        self._users_embeddings_lookup = users_embeddings_lookup
        self._not_in_lookup_embedding = not_in_lookup_embedding


    def _strip_user_profile(self, user_profile, user_embedder):
        if 'done' in user_profile and user_profile['done'] !=  'OK':
            description = ''
            user_profile = models.User(int(user_profile['user_id']))
        else:
            description = user_profile['description']
            user_profile = models.User(user_profile['id'])
        user_profile.description = description

        user = {}
        user['id'] = user_profile.id
        graphsage_embedding = self._users_embeddings_lookup.get(str(user['id']), None)
        if graphsage_embedding is None:
            graphsage_embedding = self._not_in_lookup_embedding.tolist()
        user["embedding"] = user_embedder.embed(user_profile).tolist() + graphsage_embedding
        return user


    def run(self):
        # Create output dir
        logging.info("Will output user embeddings to {}".format(self._user_embeddings_path))
        os.makedirs(self._user_embeddings_path, exist_ok=True)

        glove_embeddings = utils.load_glove_embeddings(self._embeddings_file)
        user_embedder = embeddings.UserEmbedder(glove_embeddings=glove_embeddings)

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

    logging.info("Loading dataset")

    user_profiles_path = "{}/user_profiles".format(args.input_dir)
    user_embeddings_path = "{}/user_embeddings".format(args.dataset_root)

    logging.info("Loading users embeddings graphsage lookup")
    with open(os.path.join(args.dataset_root, "datasets", "dataset0", "users_graphsage_embeddings_lookup.json")) as f:
        users_embeddings_lookup = json.load(f)

    dataset = UserProfiles(
        user_profiles_path=user_profiles_path,
        user_embeddings_path=user_embeddings_path,
        embeddings_file=args.embeddings_file,
        users_embeddings_lookup=users_embeddings_lookup,
        not_in_lookup_embedding=np.zeros(len(list(users_embeddings_lookup.values())[0]))
    )

    dataset.run()


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
