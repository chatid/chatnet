import numpy as np
# import pandas as pd
import re, string
from collections import Counter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

numeric_pat = re.compile('.*[\d].*')
caps_pat = re.compile('.*[A-Z].*')

GLOVE_VEC_FILENAME = '/Users/bhtucker/rc/chatnet/glove.twitter.27B/glove.twitter.27B.200d.txt'

# vocab = set([x[0] for x in word_counts.most_common(10050)[50:]])
# word_list = list(vocab)
# word_index = {v: ix for ix, v in enumerate(word_list)}

def get_word_index(word_counts, nb_words=15000, skip_top=None, nonembeddable=None):
    skip_top = skip_top or 10

    vocab = []
    for (ix, (w, _)) in enumerate(word_counts.most_common(nb_words + skip_top)):
        if w.startswith('$'):
            if ix < skip_top:
                skip_top += 1
            vocab.append(w)
        elif (not nonembeddable or not w in nonembeddable) and ix > skip_top:
            vocab.append(w)

    return {v: ix for ix, v in enumerate(vocab)}


class TextPrepper(object):
    def __init__(self, exclude=set(string.punctuation), pad_char=0, start_char=1, oov_char=2, index_from=3):
        self.pad_char = pad_char
        self.oov_char = oov_char
        self.start_char = start_char
        self.index_from = index_from
        self.exclude = exclude

    def cleaner(self, word):
        if not numeric_pat.match(word):
            return ''.join(ch for ch in word.lower() if ch not in self.exclude)
        if '$' in word:
            return '$price'
        if '800' in word:
            return '$phone'
        if '-' in word or caps_pat.match(word):
            return '$model'
        if 'www' in  word or 'http' in word:
            return '$web'
        else:
            return '$digit'
    
    def to_matrices(self, df, word_index, id_col='Chat Session ID', label_col='Chat Type',
                     data_col='msgs', seed=133, test_split=.2, **kwargs):

        df = df[~df[id_col].isnull()]
        ids = df[id_col]
        categories = df[label_col].unique().tolist()
        labels = df[label_col].map(lambda v: categories.index(v))
        labels = zip(ids, labels)
        X = df[data_col].tolist()

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)
        test_ix = int(len(X) * (1 - test_split))
        X_train = np.array(X[:test_ix])
        labels_train = labels[:test_ix]

        X_test = np.array(X[test_ix:])
        labels_test = labels[test_ix:]

        X_train, labels_train = self.chunk_convos(X_train, labels_train, word_index, **kwargs)
        X_test, labels_test = self.chunk_convos(X_test, labels_test, word_index, **kwargs)

        y_train = np.array([x[1] for x in labels_train])
        train_ids = [x[0] for x in labels_train]

        y_test = np.array([x[1] for x in labels_test])
        test_ids = [x[0] for x in labels_test]

        return (X_train, y_train, train_ids), (X_test, y_test, test_ids)

    def chunk_convos(self, X, labels, word_index, chunk_size=100,
                     max_dummy_ratio=2, chunk_overlap_ratio=2):
        chunk_X, chunk_labels = [], []
        skipped = 0
        for x, label in zip(X, labels):
            # x is the whole chat
            l = len(x)
            clean_x = map(self.cleaner, x)
            index_representation = [
                word_index[w] + self.index_from
                    if w in word_index
                    else self.oov_char
                    for w in clean_x
                ]

            chunk_idx = 0
            for start in (range(0, l - chunk_size, chunk_size / chunk_overlap_ratio) or [0]):
                chunk_idx += 1
                chunk = [self.start_char] + index_representation[start:start + chunk_size]
                if not chunk:
                    continue
                pad_size = (chunk_size - len(chunk) + 1) #  add a one due to start_char
                padded = [self.pad_char] * pad_size + chunk
                if sum(c == self.oov_char for c in padded) > ((chunk_size - pad_size) / max_dummy_ratio):
                    skipped += 1
                    continue
                else:
                    chunk_X.append(padded)
                    chunk_labels.append(('_'.join([label[0], str(chunk_idx)]), label[1]))
        print skipped
        return chunk_X, chunk_labels

def get_embedding_weights(word_index, index_from=3, vocab_dim=200, extra_dims=0):
    n_symbols = len(word_index) + index_from
    embedding_weights = np.zeros((n_symbols, vocab_dim + extra_dims))

    def update_weights(line):
        if line[:line.find(' ')] in word_index:
            tokens = line.split()
            embedding_weights[word_index[tokens[0]] + index_from] = \
                np.array(map(np.float, tokens[1:] + [0] * extra_dims))

    with open(GLOVE_VEC_FILENAME, 'r') as f:
        map(update_weights, f)

    return embedding_weights, n_symbols


def get_word_counts(data, tp):
    word_counts = Counter()

    def increment(word):
        word_counts[tp.cleaner(word)] += 1

    data.map(lambda r: map(increment, r))

    return word_counts

def get_nonembeddable_set(word_counts, rank_cutoff=50000):
    word_index = get_word_index(word_counts, nb_words=rank_cutoff)
    seen = set()
    with open(GLOVE_VEC_FILENAME, 'r') as f:
        for line in f:
            if line[:line.find(' ')] in word_index:
                seen.add(line[:line.find(' ')])
    return set(word_index) - seen




# invalid_glove_indices = [ix for ix, v in enumerate(embedding_weights) if np.allclose(v, 0)]

# (X_train, y_train, train_ids), (X_test, y_test, test_ids) = \
#     tp.to_matrices(s_data, s_word_index, seed=212, test_split=.08, chunk_size=100, max_dummy_ratio=1.2)



# ?????
# whitespace?




# index_dict = {
#  'yellow': 1,
#  'four': 2}

# word_vectors = {
#  'yellow': array([0.1,0.5,...,0.7]),
#  'four': array([0.2,1.2,...,0.9]),
# ...
# }


# vocab_dim = 200 # dimensionality of your word vectors
# n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)
# embedding_weights = np.zeros((n_symbols+1,vocab_dim))
# for word, index in index_dict.iteritems():
#     embedding_weights[index, :] = word_vectors[word]

# # assemble the model
# model = Sequential() # or Graph or whatever
# takes in: [3, 5, 6] seq -> [[256-vec], [256-vec], [256-vec]]
# model.add(Embedding(output_dim=rnn_dim, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
# model.add(LSTM(dense_dim, return_sequences=False))  
# model.add(Dropout(0.5))
# model.add(Dense(n_symbols, activation='softmax')) # for this is the architecture for predicting the next word, but insert your own here


