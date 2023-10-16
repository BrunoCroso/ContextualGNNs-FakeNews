#!/usr/bin/env python

#
# Tweets to trees version for FakeNews
#

import argparse
import os
import logging
import utils
import random

from dataset import FakeNewsDataset
from trees import create_trees, tree_to_dict
import json


def run(args):

    random.seed(31)

    logging.info("Loading dataset")

    dataset_pkl = "produced_data/tweets-to-trees-dataset.pkl"
    if not args.ignore_pkl and os.path.exists(dataset_pkl):
        logging.info("Loading dataset from: {}".format(dataset_pkl))
        dataset = utils.read_pickle_from_file(dataset_pkl)
    else:
        user_profiles_path = "../raw_data/user_profiles"
        user_followers_path = "../raw_data/user_followers"
        user_embeddings_path = "../raw_data/user_embeddings"
        real_news_retweets_path = "../raw_data/{}/real".format(args.website)
        fake_news_retweets_path = "../raw_data/{}/fake".format(args.website)
        dataset = FakeNewsDataset(
            user_profiles_path=user_profiles_path,
            user_followers_path=user_followers_path,
            user_embeddings_path=user_embeddings_path,
            real_news_retweets_path=real_news_retweets_path,
            fake_news_retweets_path=fake_news_retweets_path,
        )
        dataset.load(sample_probability=args.sample_probability)
        utils.write_object_to_pickle_file(dataset_pkl, dataset)

    trees_path = "produced_data/trees"
    logging.info("Writing trees to: {}".format(trees_path))
    os.makedirs(trees_path, exist_ok=True)
    for i, tree in enumerate(create_trees(dataset)):
        tree_path = os.path.join(trees_path, "trees-{}.json".format(i))
        with open(tree_path, 'w') as tree_file:
            json.dump(tree_to_dict(tree), tree_file)
            #tree_file.write(tree_to_json(tree))


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python tweets_to_trees.py")
    parser.add_argument(
        "--website",
        help="Either politifact or gossipcop",
        dest="website",
        type=str,
        default="politifact",
    )
    parser.add_argument(
        "--sample-probability",
        help="Sample probability",
        dest="sample_probability",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--ignore-dataset-pkl",
        help="Ignore the already produced pkl file of the dataset",
        dest="ignore_pkl",
        action='store_true'
    )

    args = parser.parse_args()
    run(args)
