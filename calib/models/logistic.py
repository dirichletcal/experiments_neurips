from __future__ import division

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans

import theano


class _TheanoOptimizer:
    def __init__(self, X, y, coefs_0, intercept_0):
        l_rate = theano.tensor.fscalar('l_rate')

        coefs = theano.shared(coefs_0.astype('float32'), 'coefs')
        intercept = theano.shared(intercept_0, 'intercept')
        params = [coefs, intercept]
        data = theano.shared(X.astype('float32'), 'data')
        target = theano.shared(y.astype('float32'), 'y')

        e = theano.tensor.exp(-(theano.tensor.dot(data, coefs) - intercept))
        p = 1. / (1. + e)
        cost = theano.tensor.nnet.binary_crossentropy(p, target).sum()

        gradients = theano.tensor.grad(cost, params)
        updates = [(p, p - (l_rate * g)) for p, g in zip(params, gradients)]
        self.step = theano.function([l_rate], cost, updates=updates,
                                    allow_input_downcast=True)
        self.get_params = theano.function([], params)


class LogisticRegression(BaseEstimator, RegressorMixin):
    def __init__(self, l_rate=0.01, n_iter=100):
        self.l_rate = l_rate
        self.n_iter = n_iter

    def fit(self, X, y, sample_weight=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        coefs_0 = np.random.randn(X.shape[1], 1)
        intercept_0 = np.random.randn()
        optimizer = _TheanoOptimizer(X, y, coefs_0, intercept_0)
        for iter in np.arange(self.n_iter):
            optimizer.step(self.l_rate)
        self.coefs, self.intercept = optimizer.get_params()

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        e = np.exp(-(np.dot(X, self.coefs) - self.intercept))
        return (1. / (1. + e)).flatten()
