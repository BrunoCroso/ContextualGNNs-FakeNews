#!/usr/bin/env python

#
# Preprocess the FakeNews dataset. Create a folder which contains one file 
# per tweet. Inside each tweet we have its list of retweets.
#

import argparse
import os
import logging
import utils
import random
import copy

from dataset import FakeNewsDataset
from trees import create_trees, tree_to_dict
import json


def run(args):

    random.seed(31)

    logging.info("Loading dataset")

    user_profiles_path = "../raw_data/user_profiles"
    user_followers_path = "../raw_data/user_followers"
    user_embeddings_path = "../raw_data/user_embeddings"
    real_news_retweets_path = "../raw_data/{}/real".format(args.website)
    fake_news_retweets_path = "../raw_data/{}/fake".format(args.website)
    
    output_path = "tweets1"
    os.makedirs(output_path, exist_ok=True)

    paths = { "fake": fake_news_retweets_path, "real": real_news_retweets_path }

    count = 0
    for label, path in paths.items():
        for fentry in os.scandir(path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    for path in utils.create_json_files_iterator(
                        retweets_path, sample_probability=args.sample_probability
                    ):
                        with open(path) as json_file:
                            retweets_dict = json.load(json_file)
                            retweets = retweets_dict.get("retweets", [])
                            if len(retweets) == 0:
                                continue

                            logging.debug("Loading retweets from file {}".format(path))
                            if count % 250 == 0: 
                                logging.info("{}".format(count))
                            count += 1
                            tweet_id = os.path.splitext(os.path.basename(path))[0]
                            tweet = None
                            all_retweets = []
                            for retweet in retweets:
                                if tweet is None: 
                                    tweet = copy.deepcopy(retweet["retweeted_status"])
                                all_retweets.append(copy.deepcopy(retweet))                            
                            tweet["retweets"] = all_retweets
                            tweet["label"] = label

                            output_filename = "{}/{}.json".format(output_path, tweet_id)
                            with open(output_filename, "wt") as out_file:
                                json.dump(tweet, out_file)
                        


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python dataset_preprocess.py")
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
        default=None,
    )
    parser.add_argument(
        "--ignore-dataset-pkl",
        help="Ignore the already produced pkl file of the dataset",
        dest="ignore_pkl",
        action='store_true'
    )

    args = parser.parse_args()
    run(args)
