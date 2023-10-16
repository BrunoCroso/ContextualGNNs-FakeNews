#!/usr/bin/env python

#
# Compute user labels based on their participation in astroturfing.
#

import argparse
import json
import os
import logging

class UserLabels:
    def __init__(
        self,
        real_news_retweets_path,
        fake_news_retweets_path,
        user_profiles_path,
        user_labels_path,
    ):
        self._user_labels = {}
        self._real_news_retweets_path = real_news_retweets_path
        self._fake_news_retweets_path = fake_news_retweets_path
        self._user_profiles_path = user_profiles_path
        self._user_labels_path = user_labels_path

    def run(self):
        """Load the dataset and compute labels."""
        for fentry in os.scandir(self._fake_news_retweets_path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    self._load_retweets_from_disk(retweets_path, real=False)

        for fentry in os.scandir(self._real_news_retweets_path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    self._load_retweets_from_disk(retweets_path, real=True)

        # Create output dir
        logging.info("Will output user labels to {}".format(self._user_labels_path))
        os.makedirs(self._user_labels_path, exist_ok=True)

        for user_id, (real_count, fake_count)  in self._user_labels.items(): 
            user_path = "{}/{}.json".format(self._user_labels_path, user_id)
            with open(user_path, "w") as json_file:
                logging.debug("Writing user label to file {}".format(user_path))
                user_obj = { "id": user_id, "real": real_count, "fake": fake_count }
                json.dump(user_obj, json_file)

    def _load_retweets_from_disk(self, dirpath, real):
        for fentry in os.scandir(dirpath):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    retweets_dict = json.load(json_file)
                    retweets = retweets_dict.get("retweets", [])
                    if len(retweets) == 0:
                        continue
                    logging.debug("Loading retweets from file {}".format(fentry.path))
                    for retweet in retweets:
                        self._update_tweet(retweet, real=real)

    def _update_tweet(self, tweet_dict, real):
        # load tweet user 
        if "user" in tweet_dict:
            user_id = str(tweet_dict["user"]["id"])
        elif "userid" in tweet_dict:
            user_id = str(tweet_dict["userid"])
        else:
            raise ValueError("Failed to parse user in tweet")

        if user_id not in self._user_labels: 
            self._user_labels[user_id] = (0, 0)
        x, y = self._user_labels[user_id]
        if real: 
            self._user_labels[user_id] = (x+1, y)
        else:
            self._user_labels[user_id] = (x, y+1)


def run(args):

    logging.info("Loading dataset")

    user_profiles_path = "{}/user_profiles".format(args.input_dir)
    user_labels_path = "produced_data/user_labels"
    real_news_retweets_path = "{}/{}/real".format(args.input_dir, args.website)
    fake_news_retweets_path = "{}/{}/fake".format(args.input_dir, args.website)


    dataset = UserLabels(
        user_profiles_path=user_profiles_path,
        user_labels_path=user_labels_path,
        real_news_retweets_path=real_news_retweets_path,
        fake_news_retweets_path=fake_news_retweets_path,
    )

    dataset.run()


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python compute_user_labels.py"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing the fakenewsnet dataset",
        dest="input_dir",
        type=str, 
        required=True
    )
    parser.add_argument(
        "--website",
        help="Either politifact or gossipcop",
        dest="website",
        type=str,
        default="politifact",
    )
    args = parser.parse_args()
    run(args)
