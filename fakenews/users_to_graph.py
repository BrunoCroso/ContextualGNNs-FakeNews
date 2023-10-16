#!/usr/bin/env python

import argparse
import json
import os
from typing import Dict
import models
import embeddings
import utils

import logging

import jgrapht
import pandas as pd
from tqdm import tqdm

def strip_user_profile(user_profile:Dict, embedder: embeddings.UserEmbedder) -> Dict:
    if 'done' in user_profile and user_profile['done'] !=  'OK':
        description = ''
        user_profile = models.User(int(user_profile['user_id']))
    else:
        description = user_profile['description']
        user_profile = models.User(user_profile['id'])
    user_profile.description = description

    user = {}
    user['id'] = user_profile.id
    embedding = embedder.embed(user_profile)
    for i, dimension_value in enumerate(embedding):
        user['embedding_%s' % (i)] = dimension_value
    return user

#def strip_user_profile(user_profile:Dict, embedder: embeddings.UserEmbedder) -> Dict:
#    user = {}
#    if 'done' in user_profile and user_profile['done'] !=  'OK':
#        user['id'] = int(user_profile['user_id'])
#    else:
#        user['id'] = user_profile['id']
#
#    user['protected'] = user_profile.get('protected', None)
#    user['followers_count'] = user_profile.get('followers_count', None)
#    user['friends_count'] = user_profile.get('friends_count', None)
#    user['listed_count'] = user_profile.get('listed_count', None)
#    user['favourites_count'] = user_profile.get('favourites_count', None)
#    user['verified'] = user_profile.get('verified', None)
#    user['statuses_count'] = user_profile.get('statuses_count', None)
#    user['has_extended_profile'] = user_profile.get('has_extended_profile', None)
#    user['geo_enabled'] = user_profile.get('geo_enabled', None)
#
#    return user

def build_graphs():
    with open(os.path.join(args.dataset_root, "train_user_ids.json")) as f:
        train_user_ids = json.load(f)["user_ids"]
    train_graph = build_initial_graph(train_user_ids)
    logging.info("Created train graph with {} vertices".format(train_graph.number_of_vertices))
    logging.info("Created train graph with {} edges".format(train_graph.number_of_edges))

    logging.info("Create train edges df")
    edges_df = edges_to_df(train_graph)
    logging.info("Writing train edges to pickle file")
    utils.write_object_to_pickle_file(os.path.join(args.dataset_root, "train_edges.pkl"), edges_df)
    del edges_df
    logging.info("Create train vertices df")
    vertices_df = vertices_to_df(train_graph)
    del train_graph
    logging.info("Writing train vertices to pickle file")
    utils.write_object_to_pickle_file(os.path.join(args.dataset_root, "train_vertices.pkl"), vertices_df)
    del vertices_df

    with open(os.path.join(args.dataset_root, "val_user_ids.json")) as f:
        val_user_ids = json.load(f)["user_ids"]
    val_graph = build_initial_graph(val_user_ids)
    logging.info("Created val graph with {} vertices".format(val_graph.number_of_vertices))
    logging.info("Created val graph with {} edges".format(val_graph.number_of_edges))

    logging.info("Create edges df")
    edges_df = edges_to_df(val_graph)
    logging.info("Writing val edges to pickle file")
    utils.write_object_to_pickle_file(os.path.join(args.dataset_root, "val_edges.pkl"), edges_df)
    del edges_df
    logging.info("Create val vertices df")
    vertices_df = vertices_to_df(val_graph)
    del val_graph
    logging.info("Writing val vertices to pickle file")
    utils.write_object_to_pickle_file(os.path.join(args.dataset_root, "val_vertices.pkl"), vertices_df)
    del vertices_df

    with open(os.path.join(args.dataset_root, "test_user_ids.json")) as f:
        test_user_ids = json.load(f)["user_ids"]
    test_graph = build_initial_graph(test_user_ids)
    logging.info("Created test graph with {} vertices".format(test_graph.number_of_vertices))
    logging.info("Created test graph with {} edges".format(test_graph.number_of_edges))

    logging.info("Create test edges df")
    edges_df = edges_to_df(test_graph)
    logging.info("Writing test edges to pickle file")
    utils.write_object_to_pickle_file(os.path.join(args.dataset_root, "test_edges.pkl"), edges_df)
    del edges_df
    logging.info("Create test vertices df")
    vertices_df = vertices_to_df(test_graph)
    del test_graph
    logging.info("Writing test vertices to pickle file")
    utils.write_object_to_pickle_file(os.path.join(args.dataset_root, "test_vertices.pkl"), vertices_df)
    del vertices_df

def build_initial_graph(node_ids):
    """Read user profiles from a json directory and create a first basic
    graph where users are vertices and edges correspond to 'follower' relationships.
    """
    node_ids = set(node_ids)
    logging.info("Creating graph from users")
    glove_embeddings = utils.load_glove_embeddings(args.embeddings_file)
    embedder = embeddings.UserEmbedder(glove_embeddings=glove_embeddings)

    user_profiles_path = "{}/user_profiles".format(args.input_dir)
    user_followers_path = "{}/user_followers".format(args.input_dir)

    g = jgrapht.create_graph(
        directed=True,
        allowing_self_loops=True,
        allowing_multiple_edges=True,
        any_hashable=True,
    )

    length = len(list(os.scandir(user_profiles_path)))
    pbar = tqdm(os.scandir(user_profiles_path), total=length)
    for fentry in pbar:
        pbar.set_description('File:%s'%(fentry.path))
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                user_profile = json.load(json_file)
                user = strip_user_profile(user_profile, embedder)
                if not str(user["id"]) in node_ids:
                    continue

                v = g.add_vertex(user["id"])
                g.vertex_attrs[v].update(**user)

    length = len(list(os.scandir(user_followers_path)))
    pbar = tqdm(os.scandir(user_followers_path), total=length)
    i=0
    for fentry in pbar:
        pbar.set_description('File: %s'%(fentry.path))
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                try:
                    user_followers = json.load(json_file)
                
                except:
                    i = i+1
                    continue

                if not str(user_followers["user_id"]) in node_ids:
                    continue

                if g.contains_vertex(user_followers["user_id"]) is False:
                    continue
                    user_profile = {"id": user_followers["user_id"], "description": ""}
                    user = strip_user_profile(user_profile, embedder)
                    v = g.add_vertex(user["id"])
                    g.vertex_attrs[v].update(**user)

                for follower in user_followers["followers"]:
                    if not str(follower) in node_ids:
                        continue

                    if g.contains_vertex(follower) is False:
                        continue
                        user_profile = {"id": follower, "description": ""}
                        user = strip_user_profile(user_profile, embedder)
                        v = g.add_vertex(user["id"])
                        g.vertex_attrs[v].update(**user)

                    g.add_edge(follower, user_followers["user_id"])

    print(f"Number of jsons unloaded {i}")

    return g

def edges_to_df(g):
    sources = []
    targets = []

    for e in g.edges:
        u = g.edge_source(e)
        v = g.edge_target(e)
        sources.append(u)
        targets.append(v)

    return pd.DataFrame({"source": sources, "target": targets})


def vertices_to_df(g):
    data = {}

    vertices = []
    for v in g.vertices:
        vertices.append(v)

    features = list(g.vertex_attrs[v].keys())
    for f in features:
        values = []
        for v in g.vertices:
            v = g.vertex_attrs[v][f]
            values.append(v)

        data[f] = values

    return pd.DataFrame(data, index=vertices)



def run(args):
    build_graphs()
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        epilog="Example: python users_to_graph.py --input-dir raw_data --output-file users_graph.json"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing user profiles as json files",
        dest="input_dir",
        type=str,
        default="raw_data",
    )

    parser.add_argument(
        "--dataset-root",
        help="Dataset root path",
        dest="dataset_root",
        type=str
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
