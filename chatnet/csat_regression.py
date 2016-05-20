"""
    csat_regression
    ~~~~~~~~~~~~~~~

    Makes CSAT regressions
"""


from chatnet.pipes import Pipeline

from . import logger
from collections import Counter
import math
from scipy import sparse
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

def sequence_to_csr(row):
    bow_counts = Counter(row)
    for word_ix, ct in bow_counts.iteritems():
        yield (word_ix, ct)


def create_csr_matrix(x_train, n_symbols, skip_top=3):
    """
    Given training data in form of sequences of word indices
    create sparse matrix of vocab-wide count vectors
    """
    rows = []
    cols = []
    vals = []
    for data_ix, row in enumerate(x_train):
        for word_ix, ct in sequence_to_csr(row):
            if word_ix < skip_top:
                continue
            rows.append(data_ix)
            cols.append(word_ix)
            vals.append(ct)

    return sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(len(x_train), n_symbols)
    )


def train_pca_regression(learning_data, pca_dims, probability=True, cache_size=3000, **svm_kwargs):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data

    pca = TruncatedSVD(n_components=pca_dims)
    n_symbols = max(
        np.max(X_train) + 1, np.max(X_test) + 1
    )
    logger.info("Forming CSR Matrices")
    x_train, x_test = create_csr_matrix(X_train, n_symbols), create_csr_matrix(X_test, n_symbols)
    logger.info("Starting PCA")
    pca = pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    logger.info("Starting Regression")
    l_reg = LogisticRegression.fit(x_train, y_train)
    logger.info("Scoring Regression")
    logger.info(svc.score(x_test, y_test))
    pca.n_symbols = n_symbols
    return l_reg, pca, x_train_pca, x_test_pca

def get_pca_mats(learning_data, pca):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data
    n_symbols = np.max(X_train) + 1
    x_train, x_test = create_csr_matrix(X_train, n_symbols), create_csr_matrix(X_test, n_symbols)
    if not pca:
        return x_train, x_test
    return pca.transform(x_train), pca.transform(x_test)


def split_learning_data(x_data, y_data, ids, split_ratio=.2):
    cutoff = int(math.ceil(len(x_data) * (1 - split_ratio)))
    return (x_data[:cutoff], y_data[:cutoff], ids[:cutoff]), (x_data[cutoff:], y_data[cutoff:], ids[cutoff:])

class CSAT_Regression_Pipeline(Pipeline):
    captured_kwargs = {'pca_dims', 'probability', 'cache_size', 'df'}
    persisted_attrs = {'l_reg', 'pca', 'word_index'}
    def __init__(self, *args, **kwargs):
        super kwargs = {k: v for k, v in kwargs.iteritems() if k not in self.captured_kwargs}
        super(CSAT_Pipeline, self).__init__(**super_kwargs)
        self.pca_dims = kwargs.get('pca_dims', 500)
        self.probability = kwargs.get('probability', True)
        self.cache_size = kwargs.get('cache_size', 3000)
        if 'df' in kwargs:
            self.setup(kwargs['df'])

    def run(self, **training_options):
        training_options.setdefault('pca_dims', self.pca_dims)
        training_options.setdefault('probability', self.probability)
        training_options.setdefault('cache_size', self.cache_size)
        self.l_reg, self.pca, self.x_train_pca, self.x_train_pca = train_pca_regression(self.learning_data, **training_options)

    def predict(self, new_df):
        self._set_token_data(new_df)
        self._set_learning_data(test_split = 0)
        (x, y, ids), _ = self.learning_data
        x_pca = self.pca.transform(create_csr_matrix(X, self.pca.n_symbols))
        return(self.l_reg.predict_proba(x_pca), ids)
