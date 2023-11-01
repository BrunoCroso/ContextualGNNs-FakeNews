#!/usr/bin/env python

'''
This script processes the FakeNews dataset by creating trees of retweets for each tweet 
Each tree is represented as a directed graph, where the root tweet is at the top, and subsequent
retweets are added as children. The resulting trees are saved as JSON files in the "trees" folder inside "produced_data".
'''

import argparse
import os
import logging
import utils
import random
import copy
import logging
import re
import random
import jgrapht

import models
from dataset import FakeNewsDataset
from trees import tree_to_dict
import json
from datetime import datetime 

USER_PROFILES_PATH = "../raw_data/user_profiles"
USER_FOLLOWERS_PATH = "../raw_data/user_followers"
MISSING_USER_PROFILES_PATH = "missing_user_profiles"
TREES_PATH = "trees2"

def _lookup_RT(text):
    '''
    This function receives the retweet information (text) and searches the input text for the pattern "RT @username:",
    where username is the retweet source's username. If found, it returns the username; otherwise, it returns None.
    '''

    match = re.search(r'RT\s@((\w){1,15}):', text)
    if match: 
        return match.group(1)
    return None


def _find_retweet_source(retweet, previous_retweets):
    """
    Given a retweet and all previous retweers estimate from which 
    retweet it originated.
    """

    user = retweet.user
    rt_username = _lookup_RT(retweet.text)

    # Find a tweet from the RT user
    if rt_username is not None:
        candidates = []
        for rt in previous_retweets:
            if rt.user.screenname == rt_username: 
                candidates.append(rt)

        if len(candidates) != 0:
            return max(candidates, key= lambda k: rt.user.popularity)

    # Check if we follow some of the previous users that retweeted
    candidates = []
    for rt in previous_retweets:
        if user.id in rt.user.followers:
            candidates.append(rt)

    if len(candidates) != 0:
        return max(candidates, key= lambda k: rt.user.popularity)

    # Assign to most popular based on popularity
    weights = [rt.user.popularity for rt in previous_retweets]
    return random.choices(previous_retweets, weights=weights, k=1)[0]


def load_user_from_disk(user_id):
    '''
    This function is responsible for loading user information from a JSON file on disk in a User object,
    based on the provided user ID. The loaded information includes follower count,
    list count, favorites count, status count, verification information, protection information,
    and other user profile-related details.
    '''

    #print("Looking for user {}".format(user_id))
    user = models.User(user_id)

    user_filename = "{}/{}.json".format(USER_PROFILES_PATH, user_id)
    if not os.path.exists(user_filename): 
        with open("{}/{}.json".format(MISSING_USER_PROFILES_PATH, user_id), "wt") as json_file:
            json.dump({ "id": user_id }, json_file)
        return user

    # load user from file
    with open(user_filename) as json_file:
        user_dict = json.load(json_file)

        # failed when downloading
        if "done" in user_dict and user_dict["done"] == "ERROR": 
            return user

        if str(user_dict["id"]) != user_id:
            raise ValueError(
                "Invalid userid {} in json files".format(str(user_dict["id"]))
            )

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

    user_followers_filename = "{}/{}.json".format(USER_FOLLOWERS_PATH, user_id)
    if not os.path.exists(user_followers_filename): 
        return user

    # load user followers from file
    with open("{}/{}.json".format(USER_FOLLOWERS_PATH, user_id)) as json_file:
        try:
            followers_dict = json.load(json_file)
            for follower_id in followers_dict.get("followers", []):
                user.followers.add(str(follower_id))
        except:
            pass

    return user


def create_tweet(tweet_dict, real):
    '''
    Create a Tweet object from a dictionary of tweet data.
    '''

    tweet = models.Tweet(str(tweet_dict["id"]))
    tweet.real = real
    tweet.created_at = datetime.strptime(
            tweet_dict["created_at"], "%a %b %d %H:%M:%S %z %Y"
        )
    tweet.text = tweet_dict["text"]
    
    return tweet

def get_user_id(tweet_dict):
    '''
    Extract the user ID from a dictionary of tweet data.
    '''

    if "user" in tweet_dict:
        user_id = str(tweet_dict["user"]["id"])
    elif "userid" in tweet_dict:
        user_id = str(tweet_dict["userid"])
    else:
        raise ValueError("Failed to parse user in tweet: {}".format(tweet_dict))
    return user_id

def create_tree(tweet_dict, min_retweets): 
    '''
    This function processes a dictionary containing tweet data and constructs a tree of retweets.
    The tree is represented as a directed graph where the original tweet is the root, and subsequent
    retweets are added as children. The function also validates that the minimum number of retweets
    required is met, and returns the resulting tree.
    ''' 

    real = tweet_dict["label"] == "real"
    tweet = create_tweet(tweet_dict, real=real)
    tweet.user = load_user_from_disk(get_user_id(tweet_dict))

    for retweet_dict in tweet_dict["retweets"]:
        retweet = create_tweet(retweet_dict, real=real)
        retweet.retweet_of = tweet
        tweet.retweeted_by.append(retweet)
        retweet.user = load_user_from_disk(get_user_id(retweet_dict))

    if len(tweet.retweeted_by) < min_retweets:
        return None

    retweets = sorted(
        tweet.retweeted_by, key=lambda t: t.created_at, reverse=True
    )

    tree = jgrapht.create_graph(directed=True, any_hashable=True)
    tree.add_vertex(vertex=tweet)
    tree.vertex_attrs[tweet]['delay'] = 0

    previous = []
    previous.append(tweet)

    while len(retweets) != 0: 
        cur = retweets.pop()
        tree.add_vertex(vertex=cur)

        cur_retweet_of = _find_retweet_source(cur, previous)
        tree.add_edge(cur, cur_retweet_of)

        tree.vertex_attrs[cur]['delay'] = abs((cur.created_at-cur_retweet_of.created_at).total_seconds())

        previous.append(cur)

    if tweet.real: 
        tree.graph_attrs['label'] = "real"
    else:
        tree.graph_attrs['label'] = "fake"

    return tree


def postprocess_tree(tree):
    '''
    This function takes a tree represented as a directed graph, where each vertex represents a tweet,
    and converts it into a processed graph. The processed graph retains the original structure of the tree,
    but with the following enhancements:

    - Vertices are represented as integers for improved efficiency.
    - Additional attributes are computed and added to each vertex, including user-related information such as
      'user_id', 'delay', 'followers_count', 'following_count', 'verified', 'protected', 'favourites_count',
      'listed_count', and 'statuses_count'.
    - The 'label' attribute in the graph indicates whether the content is 'real' or 'fake'.
    '''

    p_tree = jgrapht.create_graph(directed=True, any_hashable=True)

    vid = 0
    tweet_to_id = {}
    for tweet in tree.vertices:
        p_tree.add_vertex(vertex=vid)

        p_tree.vertex_attrs[vid]['user_id'] = tweet.user.id
        p_tree.vertex_attrs[vid]['delay'] = tree.vertex_attrs[tweet]['delay']
        p_tree.vertex_attrs[vid]['followers_count'] = max(len(tweet.user.followers), tweet.user.followers_count)
        p_tree.vertex_attrs[vid]['following_count'] =  max(len(tweet.user.following), tweet.user.following_count)

        for key in ['verified', 'protected', 'favourites_count', 'listed_count', 'statuses_count']:
            p_tree.vertex_attrs[vid][key] = int(getattr(tweet.user, key))

        #if tweet.user.embedding is not None:
        #    p_tree.vertex_attrs[vid]['user_profile_embedding'] = tweet.user.embedding

        tweet_to_id[tweet] = vid
        vid += 1
        
    for e in tree.edges:
        u = tree.edge_source(e)
        v = tree.edge_target(e)

        p_tree.add_edge(tweet_to_id[u], tweet_to_id[v])

    p_tree.graph_attrs['label'] = tree.graph_attrs['label']

    return p_tree


def run(args):
    '''
    Process tweet data to create trees of retweets and save them as JSON files.

    This function is responsible for processing tweet data to create trees of retweets.
    Each tree is represented as a directed graph, with the original tweet at the root and
    subsequent retweets as children. The resulting trees are saved as JSON files in the
    specified directories.
    '''
    
    random.seed(31)

    logging.info("Creating trees")
    os.makedirs(TREES_PATH, exist_ok=True)
    os.makedirs(MISSING_USER_PROFILES_PATH, exist_ok=True)
    

    count = 0
    for fentry in os.scandir(args.tweets):
        tweet_path = fentry.path
        with open(tweet_path) as json_file:
            tweet_dict = json.load(json_file)
            tree = create_tree(tweet_dict, min_retweets=8) # min_retweets original era 8. Vou alterar para ver se obtemos mais dados
                                                           # Estavam sendo gerados 124 trees antes
            if tree is not None:
                if count % 25 == 0: 
                    logging.info("{}".format(count))
                tree = postprocess_tree(tree)
                tree_path = os.path.join(TREES_PATH, "trees-{}.json".format(count))
                with open(tree_path, 'w') as tree_file:
                    json.dump(tree_to_dict(tree), tree_file)
                    #tree_file.write(tree_to_json(tree))
                count += 1        


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python create_trees.py")
    parser.add_argument(
        "--tweets",
        help="Tweets directory",
        dest="tweets",
        type=str,
        default="tweets1"
    )

    args = parser.parse_args()
    run(args)
