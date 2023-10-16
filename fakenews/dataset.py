import os
import json
import logging
import random

from datetime import datetime

import utils
from models import Tweet, User


class FakeNewsDataset:
    """The fake news dataset model."""

    def __init__(
        self,
        real_news_retweets_path,
        fake_news_retweets_path,
        user_profiles_path,
        user_followers_path,
        user_embeddings_path,
    ):
        self._users_by_id = {}
        self._users_by_username = {}
        self._tweets_by_id = {}
        self._real_news_retweets_path = real_news_retweets_path
        self._fake_news_retweets_path = fake_news_retweets_path
        self._user_profiles_path = user_profiles_path
        self._user_followers_path = user_followers_path
        self._user_embeddings_path = user_embeddings_path
        self._missing_user_profiles = 0
        self._missing_user_followers = 0
        self._missing_user_embeddings = 0
        self._loaded_count = 0

    def load(self, sample_probability=None):
        """Load the dataset."""
        if sample_probability is not None:
            logging.info("Using sample probability: {}".format(sample_probability))

        for fentry in os.scandir(self._fake_news_retweets_path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    self._load_retweets_from_disk(
                        retweets_path, sample_probability=sample_probability, real=False
                    )

        for fentry in os.scandir(self._real_news_retweets_path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    self._load_retweets_from_disk(
                        retweets_path, sample_probability=sample_probability, real=True
                    )

    def _load_retweets_from_disk(self, dirpath, sample_probability, real):
        for path in utils.create_json_files_iterator(
            dirpath, sample_probability=sample_probability
        ):
            with open(path) as json_file:
                retweets_dict = json.load(json_file)
                retweets = retweets_dict.get("retweets", [])
                if len(retweets) == 0:
                    continue
                # logging.info("Loading retweets from file {}".format(path))
                for retweet in retweets:
                    self._update_tweet(retweet, real=real)

    def _get_tweet(self, tweet_id):
        if tweet_id in self._tweets_by_id:
            tweet = self._tweets_by_id[tweet_id]
        else:
            tweet = Tweet(tweet_id)
            self._tweets_by_id[tweet_id] = tweet
        return tweet

    def _update_tweet(self, tweet_dict, real):
        self._loaded_count += 1
        if self._loaded_count % 500 == 0:
            logging.info("Loading {} tweet".format(self._loaded_count))

        tweet = self._get_tweet(str(tweet_dict["id"]))
        tweet.real = real
        tweet.created_at = datetime.strptime(
            tweet_dict["created_at"], "%a %b %d %H:%M:%S %z %Y"
        )
        tweet.text = tweet_dict["text"]

        # load tweet user
        if "user" in tweet_dict:
            user = self._update_user_from_dict(tweet_dict["user"])
            tweet.user = user
        elif "userid" in tweet_dict:
            user = self._update_user_from_dict({"id": str(tweet_dict["userid"])})
            tweet.user = user
        else:
            raise ValueError("Failed to parse user in tweet: {}".format(tweet.id))

        # load additional user info from disk
        try:
            self._update_user_from_disk(user.id)
        except FileNotFoundError:
            self._missing_user_profiles += 1
            # logging.warn("Failed to locate user {} profile".format(user.id))

        try:
            self._update_user_followers_from_disk(user.id)
        except FileNotFoundError:
            self._missing_user_followers += 1
            # logging.warn("Failed to locate user {} followers".format(user.id))

        try:
            self._update_user_embeddings_from_disk(user.id)
        except FileNotFoundError:
            self._missing_user_embeddings += 1
            # logging.warn("Failed to locate user {} embeddings".format(user.id))

        # load retweet
        if "retweeted_status" in tweet_dict:
            retweet = self._update_tweet(tweet_dict["retweeted_status"], real=real)
            tweet.retweet_of = retweet
            retweet.retweeted_by.append(tweet)

        return tweet

    def _get_user(self, user_id):
        if user_id in self._users_by_id:
            user = self._users_by_id[user_id]
        else:
            user = User(user_id)
            self._users_by_id[user_id] = user
        return user

    def _update_user_from_dict(self, user_dict):
        user = self._get_user(str(user_dict["id"]))

        if user.screenname is None and "screen_name" in user_dict:
            user.screenname = user_dict["screen_name"]
            self._users_by_username[user.screenname] = user

        for key in [
            "followers_count",
            "listed_count",
            "favourites_count",
            "statuses_count",
        ]:
            current_value = getattr(user, key)
            if current_value is None or current_value == 0:
                setattr(user, key, user_dict.get(key, 0))

        for key in [
            "verified",
            "protected",
        ]:
            current_value = getattr(user, key)
            if current_value is None or current_value is False:
                setattr(user, key, user_dict.get(key, False))

        if user.following_count is None or user.following_count == 0:
            user.following_count = user_dict.get("friends_count", 0)

        if user.description is None:
            user.description = user_dict.get("description", None)

        return user

    def _update_user_from_disk(self, user_id):
        """Read a user from disk."""
        # load user from file
        with open("{}/{}.json".format(self._user_profiles_path, user_id)) as json_file:
            user_dict = json.load(json_file)
            if str(user_dict["id"]) != user_id:
                raise ValueError(
                    "Invalid userid {} in json files".format(str(user_dict["id"]))
                )
            self._update_user_from_dict(user_dict)

    def _update_user_followers_from_disk(self, user_id):
        """Read a user's followers from disk."""
        user = self._get_user(user_id)

        # load user followers from file
        with open("{}/{}.json".format(self._user_followers_path, user_id)) as json_file:
            followers_dict = json.load(json_file)
            for follower_id in followers_dict.get("followers", []):
                user.followers.add(str(follower_id))

    def _update_user_embeddings_from_disk(self, user_id):
        """Read a user's embedding from disk."""
        user = self._get_user(user_id)

        # load user embeddings from file
        with open(
            "{}/{}.json".format(self._user_embeddings_path, user_id)
        ) as json_file:
            embeddings_dict = json.load(json_file)
            user.embedding = embeddings_dict.get("embedding", [])

    @property
    def users_by_id(self):
        return self._users_by_id

    @property
    def tweets_by_id(self):
        return self._tweets_by_id

    def __repr__(self):
        return "FakeNewsDataset(%r, %r, %r)" % (
            self._users_by_id,
            self._users_by_username,
            self._tweets_by_id,
        )
