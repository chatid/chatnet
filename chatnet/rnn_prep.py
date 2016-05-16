# -*- coding: utf-8 -*-
"""
    rnn_prep
    ~~~~~~~~
 
    Data setup prep functions for RNNs
"""
import numpy as np
from prep import get_word_index
import os

GLOVE_VEC_TEMPLATE = os.environ.get('GLOVE_VEC_TEMPLATE')

GLOVE_DIMS = {25, 50, 200}


def get_embedding_weights(word_index, index_from=3, embedding_size=200, zero_pad=0):
    n_symbols = len(word_index) + index_from
    embedding_weights = np.zeros((n_symbols, embedding_size + zero_pad))

    def update_weights(line):
        if line[:line.find(' ')] in word_index:
            tokens = line.split()
            embedding_weights[word_index[tokens[0]] + index_from] = \
                np.array(map(np.float, tokens[1:] + [0] * zero_pad))

    with open(get_vec_file(embedding_size), 'r') as f:
        map(update_weights, f)

    return embedding_weights, n_symbols


def get_vec_file(dimensions):
    return GLOVE_VEC_TEMPLATE.format(dimensions=dimensions)


def get_nonembeddable_set(word_counts, rank_cutoff=50000):
    word_index = get_word_index(word_counts, nb_words=rank_cutoff)
    seen = set()
    with open(get_vec_file(list(GLOVE_DIMS)[0]), 'r') as f:
        for line in f:
            if line[:line.find(' ')] in word_index:
                seen.add(line[:line.find(' ')])
    return set(word_index) - seen