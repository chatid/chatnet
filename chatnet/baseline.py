# -*- coding: utf-8 -*-
"""
    baseline
    ~~~~~~~~
 
    Non-NN models for comparison
"""
from . import logger
from collections import Counter
import math
from scipy import sparse
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


def sequence_to_csr(row):
    bow_counts = Counter(row)
    for word_ix, ct in bow_counts.iteritems():
        yield (word_ix, ct)


def create_csr_matrix(x_train, n_symbols):
    rows = []
    cols = []
    vals = []
    for data_ix, row in enumerate(x_train):
        for word_ix, ct in sequence_to_csr(row):
            rows.append(data_ix)
            cols.append(word_ix)
            vals.append(ct)
    
    return sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(len(x_train), n_symbols)
    )


def create_vsm_matrix(x_train, embeddings):
    rows = []
    for data_ix, row in enumerate(x_train):
        row_data = []
        for word_ix in row:
            row_data.append(embeddings[word_ix])
        rows.append(np.hstack(row_data))

    return np.array(rows)



def get_svm_baseline(learning_data, pca_dims, probability=True, cache_size=3000, **svm_kwargs):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data

    rpca = RandomizedPCA(n_components=pca_dims, whiten=True)
    n_symbols = max(
        np.max(X_train) + 1, np.max(X_test) + 1
    )
    logger.info("Forming CSR Matrices")
    x_train, x_test = create_csr_matrix(X_train, n_symbols), create_csr_matrix(X_test, n_symbols)
    logger.info("Starting PCA")
    x_train_pca = rpca.fit_transform(x_train)
    x_test_pca = rpca.transform(x_test)

    logger.info("Starting SVM")
    svc = SVC(probability=probability, cache_size=cache_size, **svm_kwargs)
    svc.fit(x_train_pca, y_train)
    logger.info("Scoring SVM")
    logger.info(svc.score(x_test_pca, y_test))
    return svc, rpca, x_train_pca, x_test_pca


def score_svm(learning_data, svc, pca):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data
    n_symbols = max(
        np.max(X_train) + 1, np.max(X_test) + 1
    )
    x_test = create_csr_matrix(X_test, n_symbols)
    x_test_pca = pca.transform(x_test)
    logger.info("Scoring SVM")
    logger.info(svc.score(x_test_pca, y_test))


def get_pca_mats(learning_data, pca):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data
    n_symbols = np.max(X_train) + 1
    x_train, x_test = create_csr_matrix(X_train, n_symbols), create_csr_matrix(X_test, n_symbols)
    x_test = create_csr_matrix(X_test, n_symbols)
    return pca.transform(x_train), pca.transform(x_test)


def split_learning_data(x_data, y_data, ids, split_ratio=.2):
    cutoff = int(math.ceil(len(x_data) * (1 - split_ratio)))
    return (x_data[:cutoff], y_data[:cutoff], ids[:cutoff]), (x_data[cutoff:], y_data[cutoff:], ids[cutoff:])

