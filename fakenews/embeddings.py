'''
This script contains the classes for the embedders to generate the user_profiles and retweets embeddings
'''

from typing import Dict, List, Union

import numpy as np

import models
import utils

from transformers import AutoModel, AutoTokenizer
import torch


class GloVeUserEmbedder:
    '''
    This class is responsible for embedding user profiles using GloVe embeddings. 
    It takes a GloVe word embeddings dictionary and an optional "not_in_vocabulary_embedding" as input.
    '''

    def __init__(self, glove_embeddings: Dict=None, not_in_vocabulary_embedding: Union[List, np.ndarray]=None):
        self.__glove_embeddings = glove_embeddings
        self.__embedding_dimensions = len(glove_embeddings[list(glove_embeddings.keys())[0]])
        if not_in_vocabulary_embedding is None:
            self.__not_in_vocabulary_embedding = np.zeros(self.__embedding_dimensions)
        else:
            self.__not_in_vocabulary_embedding = not_in_vocabulary_embedding

    def embed(self, user: models.User):
        '''
        Generates an embedding for a user's description using the provided GloVe embeddings. 
        It returns a NumPy array representing the user's embedding.
        '''
        
        embedding = np.array([])
        if self.__glove_embeddings is not None:
            tokens = utils.generate_tokens_from_text(user.description)
            tokens = [token for token in tokens if token in self.__glove_embeddings]
            word_embeddings = [self.__glove_embeddings[token] for token in tokens]
            if len(word_embeddings) != 0:
                embedding = np.mean(word_embeddings, axis=0)
            else:
                embedding = self.__not_in_vocabulary_embedding
        return embedding

class BERTweetUserEmbedder:
    '''
    This class embeds user profiles using the BERTweet model.
    It takes a BERTweet model and an optional "not_in_vocabulary_embedding" as input.
    '''

    def __init__(self, bertweet_model, not_in_vocabulary_embedding: Union[List, np.ndarray]=None):
        self.__bertweet_model = bertweet_model
        self.__embedding_dimensions = 768
        self.__tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
        if not_in_vocabulary_embedding is None:
            self.__not_in_vocabulary_embedding = np.zeros(self.__embedding_dimensions)
        else:
            self.__not_in_vocabulary_embedding = not_in_vocabulary_embedding

    def embed(self, user: models.User):
        '''
        Generates an embedding for a user's description using the BERTweet model.
        It returns a NumPy array representing the user's embedding.
        '''

        embedding = np.array([])
        if self.__bertweet_model is not None:

            tokens = torch.tensor([self.__tokenizer.encode(user.description, truncation=True)])
            with torch.no_grad():
                word_embeddings = self.__bertweet_model(tokens)[0][0]

            if len(word_embeddings) != 0:
                embedding = torch.mean(word_embeddings, axis=0).numpy()
            else:
                embedding = self.__not_in_vocabulary_embedding
        return embedding


class GloVeRetweetContentEmbedder:
    '''
    This class embeds retweet content using GloVe embeddings.
    It takes a GloVe word embeddings dictionary and an optional "not_in_vocabulary_embedding" as input.
    '''
    def __init__(self, glove_embeddings: Dict=None, not_in_vocabulary_embedding: Union[List, np.ndarray]=None):
        self.__glove_embeddings = glove_embeddings
        self.__embedding_dimensions = len(glove_embeddings[list(glove_embeddings.keys())[0]])
        if not_in_vocabulary_embedding is None:
            self.__not_in_vocabulary_embedding = np.zeros(self.__embedding_dimensions)
        else:
            self.__not_in_vocabulary_embedding = not_in_vocabulary_embedding

    def embed(self, tweet: models.Tweet):
        '''
        Generates an embedding for the text of a retweet using the provided GloVe embeddings.
        It returns a NumPy array representing the retweet's text embedding.
        '''
        embedding = np.array([])
        if self.__glove_embeddings is not None:
            tokens = utils.generate_tokens_from_text(tweet.text)
            tokens = [token for token in tokens if token in self.__glove_embeddings]
            word_embeddings = [self.__glove_embeddings[token] for token in tokens]
            if len(word_embeddings) != 0:
                embedding = np.mean(word_embeddings, axis=0)
            else:
                embedding = self.__not_in_vocabulary_embedding
        return embedding


class BERTweetRetweetContentEmbedder:
    '''
    This class embeds retweet content using the BERTweet model.
    It takes a BERTweet model and an optional "not_in_vocabulary_embedding" as input.
    '''
    def __init__(self, bertweet_model, not_in_vocabulary_embedding: Union[List, np.ndarray]=None):
        self.__bertweet_model = bertweet_model
        self.__embedding_dimensions = 768
        self.__tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
        if not_in_vocabulary_embedding is None:
            self.__not_in_vocabulary_embedding = np.zeros(self.__embedding_dimensions)
        else:
            self.__not_in_vocabulary_embedding = not_in_vocabulary_embedding

    def embed(self, tweet: models.Tweet):
        '''
        Generates an embedding for the text of a retweet using the BERTweet model.
        It returns a NumPy array representing the retweet's text embedding.
        '''

        embedding = np.array([])
        if self.__bertweet_model is not None:

            tokens = torch.tensor([self.__tokenizer.encode(tweet.text, truncation=True)])
            with torch.no_grad():
                word_embeddings = self.__bertweet_model(tokens)[0][0]

            if len(word_embeddings) != 0:
                embedding = torch.mean(word_embeddings, axis=0).numpy()
            else:
                embedding = self.__not_in_vocabulary_embedding
        return embedding